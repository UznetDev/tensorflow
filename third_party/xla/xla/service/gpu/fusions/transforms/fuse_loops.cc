/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/gpu/fusions/ir/xla_gpu_ops.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {
namespace {

using mlir::SmallVector;
using mlir::Value;
using mlir::ValueRange;
using mlir::vector::ExtractOp;

#define GEN_PASS_DEF_FUSELOOPSPASS
#include "xla/service/gpu/fusions/transforms/passes.h.inc"

bool MapsHaveTheSameDomain(const IndexingMap& map1, const IndexingMap& map2) {
  if (map1.GetDimVarsCount() != map2.GetDimVarsCount() ||
      map1.GetRangeVarsCount() != map2.GetRangeVarsCount() ||
      map1.GetConstraintsCount() != map2.GetConstraintsCount()) {
    return false;
  }
  for (auto [d1, d2] : llvm::zip(map1.GetDimVars(), map2.GetDimVars())) {
    if (d1 != d2) return false;
  }
  for (auto [r1, r2] : llvm::zip(map1.GetRangeVars(), map2.GetRangeVars())) {
    if (r1 != r2) return false;
  }
  for (auto [c1, c2] :
       llvm::zip(map1.GetConstraints(), map2.GetConstraints())) {
    if (c1 != c2) return false;
  }
  return true;
}

struct FuseLoopsPass : public impl::FuseLoopsPassBase<FuseLoopsPass> {
  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::OpBuilder builder(mlir_context);
    SmallVector<ExtractOp> extracts;
    getOperation()->walk([&](mlir::Operation* op) -> void {
      if (auto extract = mlir::dyn_cast<ExtractOp>(op)) {
        extracts.push_back(extract);
      }
    });

    for (auto extract : extracts) {
      // Check that it has the following pattern:
      // %insert_loop = { %insert = vector.insert ... }
      // %extract_loop = { %extract = vector.extract %insert_loop }
      auto extract_loop = extract->getParentOfType<LoopOp>();
      if (!extract_loop) continue;
      auto insert_loop =
          mlir::dyn_cast<LoopOp>(extract.getVector().getDefiningOp());
      if (!insert_loop) continue;
      mlir::vector::InsertOp insert;
      for (auto user : insert_loop.getRegionIterArgs().back().getUsers()) {
        if ((insert = mlir::dyn_cast<mlir::vector::InsertOp>(user))) {
          break;
        }
      }
      if (!insert) continue;

      // Are the loops fusible?
      auto extract_loop_map = extract_loop.getIndexingMap();
      auto insert_loop_map = insert_loop.getIndexingMap();
      if (!MapsHaveTheSameDomain(extract_loop_map, insert_loop_map)) continue;
      // Check dimensions come from the same op. This is technically not a
      // requirement and could be modified to handle different dim args.
      for (auto [e_dim, i_dim] :
           llvm::zip(extract_loop.getDims(), insert_loop.getDims())) {
        if (e_dim.getDefiningOp() != i_dim.getDefiningOp()) {
          continue;
        }
      }

      // Check that we are inserting into the same position that we are
      // extracting from.
      auto insert_indices = insert.getDynamicPosition();
      auto extract_indices = extract.getDynamicPosition();
      if (insert_indices.size() != extract_indices.size()) {
        continue;
      }
      for (auto [in, ex] : llvm::zip(insert_indices, extract_indices)) {
        auto in_arg = mlir::dyn_cast<mlir::BlockArgument>(in);
        auto ex_arg = mlir::dyn_cast<mlir::BlockArgument>(ex);
        if (!in_arg || !ex_arg ||
            in_arg.getArgNumber() != ex_arg.getArgNumber()) {
          continue;
        }
      }

      // All requirements have been met: fuse loops.
      // map = (...)[..] -> (insert_loop_results..., extract_loop_results...)
      auto map = insert_loop_map.GetAffineMap();
      for (auto res : extract_loop_map.GetAffineMap().getResults()) {
        map = map.insertResult(res, map.getNumResults());
      }
      IndexingMap new_map(map, insert_loop_map.GetDimVars(),
                          insert_loop_map.GetRangeVars(),
                          /*rt_vars=*/{}, insert_loop_map.GetConstraints());

      SmallVector<Value, 2> inits;
      inits.insert(inits.begin(), insert_loop.getInits().begin(),
                   insert_loop.getInits().end());
      inits.insert(inits.end(), extract_loop.getInits().begin(),
                   extract_loop.getInits().end());

      mlir::IRRewriter rewriter(builder);
      rewriter.setInsertionPointAfter(insert_loop);
      auto new_loop = rewriter.create<LoopOp>(insert_loop.getLoc(), new_map,
                                              insert_loop.getDims(), inits);
      extract.setOperand(0, insert);
      insert_loop->replaceAllUsesWith(ValueRange(new_loop.getResult(0)));
      extract_loop->replaceAllUsesWith(ValueRange(new_loop.getResult(1)));

      // Map old arguments to new arguments.
      // New args = [s0...sn, insert_loop_results..., extract_loop_results...,
      // insert_inits..., extract_inits...]
      SmallVector<Value> old_insert_args;
      SmallVector<Value> old_extract_args;
      auto new_insert_args = new_loop.getRegion().front().getArguments();
      for (auto arg : new_insert_args.take_front(new_map.GetRangeVarsCount())) {
        old_insert_args.push_back(arg);
        old_extract_args.push_back(arg);
      }
      new_insert_args = new_insert_args.drop_front(new_map.GetRangeVarsCount());
      for (auto arg :
           new_insert_args.take_front(insert_loop_map.GetNumResults())) {
        old_insert_args.push_back(arg);
      }
      new_insert_args =
          new_insert_args.drop_front(insert_loop_map.GetNumResults());
      for (auto arg :
           new_insert_args.take_front(extract_loop_map.GetNumResults())) {
        old_extract_args.push_back(arg);
      }
      new_insert_args =
          new_insert_args.drop_front(extract_loop_map.GetNumResults());
      for (auto arg :
           new_insert_args.take_front(insert_loop.getInits().size())) {
        old_insert_args.push_back(arg);
      }
      new_insert_args =
          new_insert_args.drop_front(insert_loop.getInits().size());
      for (auto arg :
           new_insert_args.take_front(extract_loop.getInits().size())) {
        old_extract_args.push_back(arg);
      }

      // Merge the loops: first insert, then extract.
      rewriter.mergeBlocks(&insert_loop.getRegion().front(),
                           &new_loop.getRegion().front(), old_insert_args);
      auto insert_yield =
          mlir::dyn_cast<YieldOp>(new_loop.getRegion().front().back());
      rewriter.mergeBlocks(&extract_loop.getRegion().front(),
                           &new_loop.getRegion().front(), old_extract_args);
      auto extract_yield =
          mlir::dyn_cast<YieldOp>(new_loop.getRegion().front().back());
      rewriter.eraseOp(insert_loop);
      rewriter.eraseOp(extract_loop);

      // Rewrite the yields.
      SmallVector<Value> results;
      results.insert(results.end(), insert_yield.getOperands().begin(),
                     insert_yield.getOperands().end());
      results.insert(results.end(), extract_yield.getOperands().begin(),
                     extract_yield.getOperands().end());
      rewriter.eraseOp(insert_yield);
      rewriter.eraseOp(extract_yield);
      rewriter.setInsertionPointAfter(&new_loop.getRegion().front().back());
      rewriter.create<YieldOp>(extract.getLoc(), results);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateFuseLoopsPass() {
  return std::make_unique<FuseLoopsPass>();
}

}  // namespace gpu
}  // namespace xla
