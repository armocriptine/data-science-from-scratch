import { randomUUID } from "crypto";
import { readFileSync, writeFileSync } from "fs";
import { Matrix } from "../../matrix/Matrix";
import { MatrixLayer } from "../layers/Matrix";
import { ParameterNode } from "../nodes/Parameter";
import { InputLayer } from "../layers/Input";
import { Vector } from "../../vector/Vector";
import { InputNode } from "../nodes/Input";

export abstract class NeuralNetwork {
  constructor(
    public readonly inputLayer: MatrixLayer<InputLayer>,
    public readonly outputLayer: MatrixLayer
  ) {}

  private _cachedParameterList: ParameterNode[] | null = null;

  public predict(predictor: Matrix, sessionId = randomUUID(), training = false): Matrix {
    for (const [i, rows] of this.inputLayer.rows.entries()) {
      for (const [j, inputNode] of rows.outputNodes.entries()) {
        (inputNode as InputNode).input = predictor.at(i, j);
      }
    }

    const output = this.outputLayer.rows.map((row) =>
      row.outputNodes.map((n) => n.activate(sessionId, training))
    );

    return Matrix.fromRaw(output);
  }

  public get parameters(): ParameterNode[] {
    if (!this._cachedParameterList) {
      this._cachedParameterList = this.outputLayer.rows.flatMap((row) =>
        row.outputNodes.flatMap((n) => n.getAllParametersInPath())
      );
    }
    return this._cachedParameterList;
  }

  public loadParameters(filePath: string) {
    const params = JSON.parse(readFileSync(filePath).toString());

    for (const [i, p] of this.parameters.entries()) {
      p.value = params[i];
    }
  }

  public saveParameters(filePath: string) {
    writeFileSync(
      filePath,
      JSON.stringify(this.parameters.map((p) => p.value))
    );
  }

  public setLossGradient(lossGradient: Matrix): void {
    for (const [i, row] of this.outputLayer.rows.entries()) {
      for (const [j, node] of row.outputNodes.entries()) {
        node.setLossGradient(lossGradient.at(i, j));
      }
    }
  }
}
