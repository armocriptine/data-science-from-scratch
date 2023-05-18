import { Matrix } from "../../matrix/Matrix";
import { ActivationFunc } from "../functions/Function";
import { DecoderBlock } from "../layers/DecoderBlock";
import { EncoderBlock } from "../layers/EncoderBlock";
import { InputLayer } from "../layers/Input";
import { MatrixLayer } from "../layers/Matrix";
import { SoftmaxLayer } from "../layers/Softmax";
import { NeuralNetwork } from "./NeuralNetwork";

export class Transformer extends NeuralNetwork {
  constructor(params: {
    encoder: {
      inputWidth: number;
      inputLength: number;
      count: number;
      attentionHeads: number;
      softmaxTemp: number;
      projectedKeyQuerySize: number;
      projectedValueSize: number;
      feedforwardSize: number;
      feedforwardActivationFunc: ActivationFunc;
      dropoutRate: number;
    };
    decoder: {
      inputWidth: number;
      inputLength: number;
      count: number;
      attentionHeads: number;
      softmaxTemp: number;
      projectedKeyQuerySize: number;
      projectedValueSize: number;
      feedforwardSize: number;
      feedforwardActivationFunc: ActivationFunc;
      unembedder?: (input: MatrixLayer) => MatrixLayer;
      outputTemp: number;
      dropoutRate: number;
    };
  }) {
    // ----- encoder side
    const encoderInput = new MatrixLayer({
      rows: Array.from({ length: params.encoder.inputLength }).map(
        () => new InputLayer(params.encoder.inputWidth)
      ),
    });

    let lastEncoder: MatrixLayer = encoderInput;

    for (let i = 0; i < params.encoder.count; i++) {
      lastEncoder = new EncoderBlock({
        input: lastEncoder,
        attentionHeads: params.encoder.attentionHeads,
        softmaxTemp: params.encoder.softmaxTemp,
        projectedKeyQuerySize: params.encoder.projectedKeyQuerySize,
        projectedValueSize: params.encoder.projectedValueSize,
        feedforwardSize: params.encoder.feedforwardSize,
        feedforwardActivationFunc: params.encoder.feedforwardActivationFunc,
        dropoutRate: params.encoder.dropoutRate,
      });
    }

    // ----- decoder side
    const decoderInput = new MatrixLayer({
      rows: Array.from({ length: params.decoder.inputLength }).map(
        () => new InputLayer(params.decoder.inputWidth)
      ),
    });

    let lastDecoder: MatrixLayer = decoderInput;

    for (let i = 0; i < params.decoder.count; i++) {
      lastDecoder = new DecoderBlock({
        input: lastDecoder,
        encoder: lastEncoder,
        attentionHeads: params.decoder.attentionHeads,
        softmaxTemp: params.decoder.softmaxTemp,
        projectedKeyQuerySize: params.decoder.projectedKeyQuerySize,
        projectedValueSize: params.decoder.projectedValueSize,
        feedforwardSize: params.decoder.feedforwardSize,
        feedforwardActivationFunc: params.decoder.feedforwardActivationFunc,
        dropoutRate: params.decoder.dropoutRate,
      });
    }

    const logits = params.decoder.unembedder?.(lastDecoder) ?? lastDecoder;

    // softmax after decode
    const softmax = new MatrixLayer({
      rows: logits.rows.map(
        (row, i) => new SoftmaxLayer(row, params.decoder.outputTemp, "final_" + i)
      ),
    });

    super(
      encoderInput.concatRow(decoderInput) as MatrixLayer<InputLayer>,
      softmax
    );
  }
}
