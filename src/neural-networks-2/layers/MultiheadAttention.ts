import { UniformDistribution } from "../../prob/distributions/Uniform";
import { Vector } from "../../vector/Vector";
import { AttentionLayer } from "./Attention";
import { MatMulLayer } from "./MatMul";
import { MatrixLayer } from "./Matrix";
import { ParameterLayer } from "./Parameter";

export class MultiheadAttentionLayer extends MatrixLayer {
  constructor(params: {
    key: MatrixLayer;
    query: MatrixLayer;
    value: MatrixLayer;
    attentionHeads: number;
    softmaxTemp: number;
    projectedKeyQuerySize: number;
    projectedValueSize: number;
    masked: boolean;
  }) {
    const attentions = Array.from({ length: params.attentionHeads }).map(
      () =>
        new AttentionLayer({
          key: params.key,
          query: params.query,
          value: params.value,
          projectedKeyQuerySize: params.projectedKeyQuerySize,
          projectedValueSize: params.projectedValueSize,
          softmaxTemp: params.softmaxTemp,
          masked: params.masked,
        })
    );

    const concatSelfAttentions = attentions.reduce((prev, x) =>
      prev.concatCol(x)
    );

    // xavier initialization
    const attProjWeightBound = Math.sqrt(
      6 /
        (params.attentionHeads * params.projectedValueSize + params.value.width)
    );
    const attProjWeightInit = new UniformDistribution(
      -attProjWeightBound,
      attProjWeightBound
    );
    const attentionProjectionMatrix = new MatrixLayer({
      rows: Array.from({
        length: params.attentionHeads * params.projectedValueSize,
      }).map(
        () =>
          new ParameterLayer(
            params.value.width,
            Vector.from(attProjWeightInit, params.value.width)
          )
      ),
    });

    const projectedAttention = new MatMulLayer(
      concatSelfAttentions,
      attentionProjectionMatrix
    );

    super({ rows: projectedAttention.rows });
  }
}
