import { Layer, SimpleLayer } from "./Layer";

export class MatrixLayer<T extends SimpleLayer = SimpleLayer> extends Layer {
  constructor(private readonly rowsOrCols: { rows: T[] } | { cols: T[] }) {
    super();
  }

  public get height(): number {
    return "rows" in this.rowsOrCols
      ? this.rowsOrCols.rows.length
      : this.rowsOrCols.cols[0].outputNodes.length;
  }

  public get width(): number {
    return "rows" in this.rowsOrCols
      ? this.rowsOrCols.rows[0].outputNodes.length
      : this.rowsOrCols.cols.length;
  }

  public at(
    row: number,
    col: number
  ): T extends SimpleLayer<infer U> ? U : never {
    return "rows" in this.rowsOrCols
      ? this.rowsOrCols.rows[row].outputNodes[col]
      : (this.rowsOrCols.cols[col].outputNodes[row] as any);
  }

  public get rows(): SimpleLayer[] {
    return "rows" in this.rowsOrCols
      ? this.rowsOrCols.rows
      : Array.from({ length: this.height }).map(
          (_, i) =>
            new SimpleLayer(
              (this.rowsOrCols as { cols: SimpleLayer[] }).cols.map(
                (col) => col.outputNodes[i]
              )
            )
        );
  }

  public get cols(): SimpleLayer[] {
    return "cols" in this.rowsOrCols
      ? this.rowsOrCols.cols
      : Array.from({ length: this.width }).map(
          (_, i) =>
            new SimpleLayer(
              (this.rowsOrCols as { rows: SimpleLayer[] }).rows.map(
                (row) => row.outputNodes[i]
              )
            )
        );
  }

  public get transpose(): MatrixLayer {
    return new MatrixLayer(
      "rows" in this.rowsOrCols
        ? { cols: this.rowsOrCols.rows }
        : { rows: this.rowsOrCols.cols }
    );
  }

  public concatCol(next: MatrixLayer): MatrixLayer {
    return new MatrixLayer({ cols: [...this.cols, ...next.cols] });
  }

  public concatRow(next: MatrixLayer): MatrixLayer {
    return new MatrixLayer({ rows: [...this.rows, ...next.rows] });
  }
}
