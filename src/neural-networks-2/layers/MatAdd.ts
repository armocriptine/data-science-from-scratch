import { AddNode } from "../nodes/Add";
import { SimpleLayer } from "./Layer";
import { MatrixLayer } from "./Matrix";

export class MatAddLayer extends MatrixLayer {
  constructor(
    public readonly left: MatrixLayer,
    public readonly right: MatrixLayer
  ) {
    super({
      rows: left.rows.map(
        (leftRow, i) =>
          new SimpleLayer(
            leftRow.outputNodes.map(
              (leftNode, j) =>
                new AddNode(leftNode, right.rows[i].outputNodes[j])
            )
          )
      ),
    });
  }
}
