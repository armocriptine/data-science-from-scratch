import { MatAddLayer } from "./MatAdd";
import { MatrixLayer } from "./Matrix";
import { NormLayer } from "./Norm";

export class TransformerAddAndNormLayer extends MatrixLayer {
  constructor(params: { input1: MatrixLayer; input2: MatrixLayer }) {
    const add = new MatAddLayer(params.input1, params.input2);

    const norm = new MatrixLayer({
      rows: add.rows.map((row) => new NormLayer(row)),
    });

    super({ rows: norm.rows });
  }
}
