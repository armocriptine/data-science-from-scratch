import { MultiplyNode } from "../nodes/Multiply";
import { SumNode } from "../nodes/Sum";
import { SimpleLayer } from "./Layer";
import { MatrixLayer } from "./Matrix";

export class MatMulLayer extends MatrixLayer {
  constructor(
    public readonly left: MatrixLayer,
    public readonly right: MatrixLayer
  ) {
    const leftRows = left.rows;
    const rightCols = right.cols;

    const rows = leftRows.map(
      (leftRow) =>
        // dot product
        new SimpleLayer(
          rightCols.map(
            (rightCol) =>
              new SumNode(
                rightCol.outputNodes.map(
                  (rightNode, i) =>
                    new MultiplyNode(leftRow.outputNodes[i], rightNode)
                )
              )
          )
        )
    );

    super({ rows });
  }
}
