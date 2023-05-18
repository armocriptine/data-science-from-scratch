import { Vector } from "../vector/Vector";
import { Matrix } from "./Matrix";

export class MatrixWithHeader extends Matrix {
  constructor(
    public readonly rows: readonly Vector[],
    private readonly rowHeaders: readonly string[],
    private readonly colHeaders: readonly string[]
  ) {
    super(rows);
  }

  public static fromColsWithHeader(
    vectors: Vector[],
    rowHeaders: string[],
    colHeaders: string[]
  ): MatrixWithHeader {
    return new MatrixWithHeader(
      Matrix.fromCols(vectors).rows,
      rowHeaders,
      colHeaders
    );
  }

  public get pretty(): string {
    return [
      [" ", ...this.colHeaders].join("\t"),
      ...this.rows.map((v, r) =>
        [this.rowHeaders[r], ...v.entries.map((e) => e.toFixed(2))].join("\t")
      ),
    ].join("\n");
  }
}
