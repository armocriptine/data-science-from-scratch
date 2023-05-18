import { MultiplyNode } from "../nodes/Multiply";
import { SimpleLayer } from "./Layer";
import { MatrixLayer } from "./Matrix";

export class ElementWiseMulLayer extends MatrixLayer {
  constructor(
    public readonly left: MatrixLayer,
    public readonly right: MatrixLayer
  ) {
    super({
      rows: left.rows.map(
        (row, i) =>
          new SimpleLayer(
            row.outputNodes.map((n, j) => new MultiplyNode(n, right.at(i, j)))
          )
      ),
    });
  }
}
