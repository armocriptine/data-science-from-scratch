import { max, square, sum } from "../utils/helpers";
import { vecSum, Vector } from "../vector/Vector";

export class Matrix {
  constructor(public readonly rows: readonly Vector[]) {}

  public get width(): number {
    return this.rows[0].count;
  }

  public get height(): number {
    return this.rows.length;
  }

  public get raw(): number[][] {
    return this.rows.map((v) => v.entries) as number[][];
  }

  public get diag(): number[] {
    return this.rows.map((v, i) => v.entries[i]);
  }

  public get pretty(): string {
    return this.rows
      .map((v) => v.entries.map((e) => e.toFixed(2)).join("\t"))
      .join("\n");
  }

  public get cols(): Vector[] {
    return new Array(this.width)
      .fill(0)
      .map((_, i) => new Vector(this.rows.map((r) => r.entries[i])));
  }

  public get isUpperTriangular(): boolean {
    return this.rows.every((r, i) =>
      r.entries.slice(0, i).every((x) => x === 0)
    );
  }

  public isWeakUpperTriangular(threshold: number): boolean {
    return this.rows.every((r, i) =>
      r.entries.slice(0, i).every((x) => Math.abs(x) < threshold)
    );
  }

  public isWeakDiagonal(threshold: number): boolean {
    return this.rows.every((r, i) =>
      r.entries.every((x, j) => i === j || Math.abs(x) < threshold)
    );
  }

  public at(row: number, col: number): number {
    return this.rows[row]?.entries[col] ?? 0;
  }

  public static fromRows(vectors: Vector[]): Matrix {
    return Matrix.fromRaw(vectors.map((v) => v.entries) as number[][]);
  }

  public static fromCols(vectors: Vector[]): Matrix {
    return Matrix.fromRows(vectors).transpose;
  }

  public static fromStr(str: string): Matrix {
    return Matrix.fromRows(
      str
        .trim()
        .split("\n")
        .map(
          (l) =>
            new Vector(
              l
                .trim()
                .replace(/\s+/g, " ")
                .split(" ")
                .map((x) => parseInt(x))
            )
        )
    );
  }

  public static fromRaw(entries: number[][]): Matrix {
    return new Matrix(entries.map((e) => new Vector(e)));
  }

  public static diag(vector: Vector): Matrix {
    return Matrix.fromRaw(
      vector.entries.map((e, i) => {
        const row = new Array(vector.count).fill(0);
        row[i] = e;
        return row;
      })
    );
  }

  public static identity(size: number): Matrix {
    return Matrix.diag(Vector.repeat(1, size));
  }

  public get isSquare(): boolean {
    return this.width === this.height;
  }

  public get transpose(): Matrix {
    return Matrix.fromRows(this.cols);
  }

  public get adjoint(): Matrix {
    return Matrix.fromRaw(
      new Array(this.height)
        .fill(0)
        .map((_, r) =>
          new Array(this.width).fill(0).map((_, c) => this.cofactor(r, c))
        )
    ).transpose;
  }

  public get inverse(): Matrix {
    if (this.height === 1 && this.width === 1) {
      return Matrix.fromRaw([[1 / this.at(0, 0)]]);
    }

    return this.adjoint.times(1 / this.det);
  }

  public get det(): number {
    if (this.isSquare) {
      if (this.width === 1) {
        return this.rows[0].entries[0];
      }

      return new Array(this.width)
        .fill(0)
        .map((_, c) => this.at(0, c) * this.cofactor(0, c))
        .reduce(sum);
    }

    return Number.NaN;
  }

  public get frobeniusNorm(): number {
    return Math.sqrt(
      this.rows
        .map((r) => r.euclideanNorm)
        .map(square)
        .reduce(sum)
    );
  }

  public removeRow(row: number): Matrix {
    const rows = [...this.rows];
    rows.splice(row, 1);
    return Matrix.fromRows(rows);
  }

  public removeCol(col: number): Matrix {
    return this.transpose.removeRow(col).transpose;
  }

  public truncateRow(rowCount: number): Matrix {
    return Matrix.fromRows(this.rows.slice(0, rowCount));
  }

  public truncateCol(colCount: number): Matrix {
    return Matrix.fromCols(this.cols.slice(0, colCount));
  }

  public replaceCol(col: number, newCol: Vector): Matrix {
    const cols = this.cols;
    cols[col] = newCol;
    return Matrix.fromCols(cols);
  }

  public minor(row: number, col: number): number {
    return this.removeRow(row).removeCol(col).det;
  }

  public cofactor(row: number, col: number): number {
    return Math.pow(-1, row + col) * this.minor(row, col);
  }

  public add(another: Matrix): Matrix {
    return Matrix.fromRows(this.rows.map((r, i) => another.rows[i].add(r)));
  }

  public subtract(another: Matrix): Matrix {
    return this.add(another.neg);
  }

  public times(scalar: number): Matrix {
    return Matrix.fromRows(this.rows.map((r) => r.times(scalar)));
  }

  public get neg(): Matrix {
    return this.times(-1);
  }

  public multiply(another: Matrix): Matrix {
    const anotherCols = another.cols;
    return Matrix.fromRaw(
      this.rows.map((r) => anotherCols.map((c) => r.dot(c)))
    );
  }

  public qr(): { q: Matrix; r: Matrix } {
    // Gram-Schmidt method
    const cols = this.cols;
    const us: Vector[] = [];

    for (const col of cols) {
      us.push(
        col.subtract(
          us
            .map((u) => col.projectOnto(u))
            .reduce(vecSum, Vector.zero(this.height))
        )
      );
    }

    const es = us.map((u) => u.times(1 / u.euclideanNorm));

    const q = Matrix.fromCols(es);
    const r = Matrix.fromCols(
      cols.map((col, i) =>
        new Vector(
          new Array(i + 1).fill(0).map((_, j) => col.dot(es[j]))
        ).padEnd(0, this.width)
      )
    );

    return { q, r };
  }

  public lu(): { l: Matrix; u: Matrix } {
    // Doolittle decomposition
    const l: number[][] = new Array(this.height)
      .fill(0)
      .map(() => new Array(this.width).fill(0));
    const u: number[][] = new Array(this.height)
      .fill(0)
      .map(() => new Array(this.width).fill(0));

    for (let i = 0; i < this.height; i++) {
      for (let j = 0; j < this.height; j++) {
        if (j >= i) {
          if (i === 0) {
            u[i][j] = this.at(i, j);
          } else {
            u[i][j] =
              this.at(i, j) -
              new Array(i)
                .fill(0)
                .map((_, k) => l[i][k] * u[k][j])
                .reduce(sum);
          }
        }

        if (i >= j) {
          if (j === 0) {
            l[i][j] = this.at(i, j) / u[j][j];
          } else {
            l[i][j] =
              (this.at(i, j) -
                new Array(j)
                  .fill(0)
                  .map((_, k) => l[i][k] * u[k][j])
                  .reduce(sum)) /
              u[j][j];
          }
        }
      }
    }

    return { u: Matrix.fromRaw(u), l: Matrix.fromRaw(l) };
  }

  public eigensByQr(): { value: number; vector: Vector }[] {
    // QR method
    let a: Matrix = this;
    let s = Matrix.identity(this.height);

    for (let i = 0; i < 10000; i++) {
      const { q, r } = a.qr();
      a = r.multiply(q);
      s = s.multiply(q);

      if (a.isWeakDiagonal(0.001)) break;
    }

    const eigenvectors = s.cols;
    return a.diag
      .sort((a, b) => Math.abs(b) - Math.abs(a))
      .map((value, i) => ({
        value,
        vector: eigenvectors[i].times(
          1 / (eigenvectors[i].entries.at(-1) ?? 1)
        ),
      }));
  }

  public eigenDecompose(): { u: Matrix; lambda: Matrix } {
    const eigs = this.eigensByQr();

    return {
      u: Matrix.fromCols(eigs.map((eig) => eig.vector.normalize)),
      lambda: Matrix.diag(new Vector(eigs.map((eig) => eig.value))),
    };
  }

  public singularValueDecompose(): { u: Matrix; v: Matrix; sigma: Matrix } {
    const eigs = this.multiply(this.transpose).eigensByQr();

    const u = Matrix.fromCols(eigs.map((eig) => eig.vector.normalize));
    const sigma = Matrix.diag(
      new Vector(eigs.map((eig) => Math.sqrt(eig.value)))
    );

    const v = Matrix.fromCols(
      u.cols.map(
        (col, i) =>
          this.transpose.multiply(col.asCol).times(1 / sigma.at(i, i)).cols[0]
      )
    );

    return {
      u,
      v,
      sigma,
    };
  }

  public principalComponentAnalyze(rank: number): Matrix {
    const { u, v, sigma } = this.singularValueDecompose();

    return u
      .truncateCol(rank)
      .multiply(sigma.truncateCol(rank).truncateRow(rank))
      .multiply(v.transpose.truncateRow(rank));
  }

  public concatCols(next: Matrix): Matrix {
    return Matrix.fromCols([...this.cols, ...next.cols]);
  }

  public concatRows(next: Matrix): Matrix {
    return Matrix.fromRows([...this.rows, ...next.rows]);
  }
}
