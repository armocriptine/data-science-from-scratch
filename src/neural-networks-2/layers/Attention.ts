import { Matrix } from "../../matrix/Matrix";
import { UniformDistribution } from "../../prob/distributions/Uniform";
import { Vector } from "../../vector/Vector";
import { ConstantNode } from "../nodes/Constant";
import { ElementWiseMulLayer } from "./EntryWiseMul";
import { SimpleLayer } from "./Layer";
import { MatMulLayer } from "./MatMul";
import { MatrixLayer } from "./Matrix";
import { ParameterLayer } from "./Parameter";
import { SoftmaxLayer } from "./Softmax";

export class AttentionLayer extends MatrixLayer {
  constructor(params: {
    key: MatrixLayer;
    query: MatrixLayer;
    value: MatrixLayer;
    projectedKeyQuerySize: number;
    projectedValueSize: number;
    masked: boolean;
    softmaxTemp: number;
  }) {
    // xavier initialization
    const keyWeightBound = Math.sqrt(6 / (params.key.width + params.projectedKeyQuerySize))
    const queryWeightBound = Math.sqrt(6 / (params.query.width + params.projectedKeyQuerySize))
    const valueWeightBound = Math.sqrt(6 / (params.value.width + params.projectedValueSize))
    
    const keyWeightInit = new UniformDistribution(-keyWeightBound, keyWeightBound);
    const queryWeightInit = new UniformDistribution(-queryWeightBound, queryWeightBound);
    const valueWeightInit = new UniformDistribution(-valueWeightBound, valueWeightBound);

    const keyWeight = new MatrixLayer({
      rows: Array.from({ length: params.key.width }).map(
        () =>
          new ParameterLayer(
            params.projectedKeyQuerySize,
            Vector.from(keyWeightInit, params.projectedKeyQuerySize)
          )
      ),
    });
    const queryWeight = new MatrixLayer({
      rows: Array.from({ length: params.query.width }).map(
        () =>
          new ParameterLayer(
            params.projectedKeyQuerySize,
            Vector.from(queryWeightInit, params.projectedKeyQuerySize)
          )
      ),
    });
    const valueWeight = new MatrixLayer({
      rows: Array.from({ length: params.value.width }).map(
        () =>
          new ParameterLayer(
            params.projectedValueSize,
            Vector.from(valueWeightInit, params.projectedValueSize)
          )
      ),
    });

    const projectedKey = new MatMulLayer(params.key, keyWeight);
    const projectedQuery = new MatMulLayer(params.query, queryWeight);
    const projectedValue = new MatMulLayer(params.value, valueWeight);

    const similarity = new MatMulLayer(projectedQuery, projectedKey.transpose);
    const scaledSimilarity = new ElementWiseMulLayer(
      similarity,
      new MatrixLayer({
        rows: Array.from({ length: similarity.height }).map(
          () =>
            new SimpleLayer(
              Array.from({ length: similarity.width }).map(
                () => new ConstantNode(1 / Math.sqrt(params.key.width))
              )
            )
        ),
      })
    );

    let masked: MatrixLayer | null = null;
    if (params.masked) {
      masked = new MatrixLayer({
        rows: scaledSimilarity.rows.map(
          (row, i) =>
            new SimpleLayer(
              row.outputNodes.map((n, j) => {
                if (j <= i) {
                  return n;
                } else {
                  n.outgoingNodes.push(new ConstantNode(0));
                  return new ConstantNode(Number.NEGATIVE_INFINITY);
                }
              })
            )
        ),
      });
    }

    const softmax = new MatrixLayer({
      rows: (masked ?? scaledSimilarity).rows.map(
        (row) => new SoftmaxLayer(row, params.softmaxTemp, "attention")
      ),
    });

    const attendedValue = new MatMulLayer(softmax, projectedValue);

    super({ rows: attendedValue.rows });
  }
}
