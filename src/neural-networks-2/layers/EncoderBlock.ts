import { ActivationFunc } from "../functions/Function";
import { DropoutNode } from "../nodes/Dropout";
import { SimpleLayer } from "./Layer";
import { MatrixLayer } from "./Matrix";
import { MultiheadAttentionLayer } from "./MultiheadAttention";
import { TransformerAddAndNormLayer } from "./TransformerAddAndNormLayer";
import { TransformerFeedforwardLayer } from "./TransformerFeedforwardLayer";

export class EncoderBlock extends MatrixLayer {
  constructor(params: {
    input: MatrixLayer;
    attentionHeads: number;
    softmaxTemp: number;
    projectedKeyQuerySize: number;
    projectedValueSize: number;
    feedforwardSize: number;
    feedforwardActivationFunc: ActivationFunc;
    dropoutRate: number;
  }) {
    // ----- attention subblock
    const multiheadAtt = new MultiheadAttentionLayer({
      key: params.input,
      query: params.input,
      value: params.input,
      attentionHeads: params.attentionHeads,
      softmaxTemp: params.softmaxTemp,
      projectedKeyQuerySize: params.projectedKeyQuerySize,
      projectedValueSize: params.projectedValueSize,
      masked: false,
    });

    const attDropout = new MatrixLayer({
      rows: multiheadAtt.rows.map(
        (row) =>
          new SimpleLayer(
            row.outputNodes.map((n) => new DropoutNode(n, params.dropoutRate))
          )
      ),
    });

    const attAddAndNorm = new TransformerAddAndNormLayer({
      input1: attDropout,
      input2: params.input,
    });

    // ----- feedforward subblock
    const feedforward = new TransformerFeedforwardLayer({
      input: attAddAndNorm,
      size: params.feedforwardSize,
      activationFunc: params.feedforwardActivationFunc,
    });

    const ffDropout = new MatrixLayer({
      rows: feedforward.rows.map(
        (row) =>
          new SimpleLayer(
            row.outputNodes.map((n) => new DropoutNode(n, params.dropoutRate))
          )
      ),
    });

    const ffAddAndNorm = new TransformerAddAndNormLayer({
      input1: ffDropout,
      input2: attAddAndNorm,
    });

    super({ rows: ffAddAndNorm.rows });
  }
}
