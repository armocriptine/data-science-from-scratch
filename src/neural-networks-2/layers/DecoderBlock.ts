import { ActivationFunc } from "../functions/Function";
import { DropoutNode } from "../nodes/Dropout";
import { SimpleLayer } from "./Layer";
import { MatrixLayer } from "./Matrix";
import { MultiheadAttentionLayer } from "./MultiheadAttention";
import { TransformerAddAndNormLayer } from "./TransformerAddAndNormLayer";
import { TransformerFeedforwardLayer } from "./TransformerFeedforwardLayer";

export class DecoderBlock extends MatrixLayer {
  constructor(params: {
    input: MatrixLayer;
    encoder: MatrixLayer;
    attentionHeads: number;
    softmaxTemp: number;
    projectedKeyQuerySize: number;
    projectedValueSize: number;
    feedforwardSize: number;
    feedforwardActivationFunc: ActivationFunc;
    dropoutRate: number;
  }) {
    // ----- masked attention subblock
    const maskedAtt = new MultiheadAttentionLayer({
      key: params.input,
      query: params.input,
      value: params.input,
      attentionHeads: params.attentionHeads,
      softmaxTemp: params.softmaxTemp,
      projectedKeyQuerySize: params.projectedKeyQuerySize,
      projectedValueSize: params.projectedValueSize,
      masked: true,
    });

    const attDropout = new MatrixLayer({
      rows: maskedAtt.rows.map(
        (row) =>
          new SimpleLayer(row.outputNodes.map((n) => new DropoutNode(n, params.dropoutRate)))
      ),
    });

    const attAddAndNorm = new TransformerAddAndNormLayer({
      input1: attDropout,
      input2: params.input,
    });

    // ----- (middle) attention subblock
    const midAtt = new MultiheadAttentionLayer({
      key: params.encoder,
      query: attAddAndNorm,
      value: params.encoder,
      attentionHeads: params.attentionHeads,
      softmaxTemp: params.softmaxTemp,
      projectedKeyQuerySize: params.projectedKeyQuerySize,
      projectedValueSize: params.projectedValueSize,
      masked: false,
    });

    const midAttDropout = new MatrixLayer({
      rows: midAtt.rows.map(
        (row) =>
          new SimpleLayer(row.outputNodes.map((n) => new DropoutNode(n, params.dropoutRate)))
      ),
    });

    const midAddAndNorm = new TransformerAddAndNormLayer({
      input1: midAttDropout,
      input2: attAddAndNorm,
    });

    // ----- feedforward subblock
    const feedforward = new TransformerFeedforwardLayer({
      input: midAddAndNorm,
      size: params.feedforwardSize,
      activationFunc: params.feedforwardActivationFunc,
    });

    const ffDropout = new MatrixLayer({
      rows: feedforward.rows.map(
        (row) =>
          new SimpleLayer(row.outputNodes.map((n) => new DropoutNode(n, params.dropoutRate)))
      ),
    });

    const ffAddAndNorm = new TransformerAddAndNormLayer({
      input1: ffDropout,
      input2: midAddAndNorm,
    });

    super({ rows: ffAddAndNorm.rows });
  }
}
