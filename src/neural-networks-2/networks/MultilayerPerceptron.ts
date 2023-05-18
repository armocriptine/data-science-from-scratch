import { Matrix } from "../../matrix/Matrix";
import { ActivationFunc } from "../functions/Function";
import { ActivationLayer } from "../layers/Activation";
import { DenseLinearLayer } from "../layers/DenseLinearLayer";
import { InputLayer } from "../layers/Input";
import { SimpleLayer } from "../layers/Layer";
import { MatrixLayer } from "../layers/Matrix";
import { NormLayer } from "../layers/Norm";
import { SoftmaxLayer } from "../layers/Softmax";
import { NeuralNetwork } from "./NeuralNetwork";

export class MultilayerPerceptron extends NeuralNetwork {
  constructor(params: {
    inputWidth: number;
    hiddenWidth: number[];
    outputWidth: number;
    hiddenActivationFuncs: ActivationFunc[];
    outputActivationFunc:
      | ActivationFunc<number>
      | {
          type: "softmax";
          temperature: number;
        };
    parameterInitializer: () => number;
    normalizationLayer: boolean;
  }) {
    const input = new InputLayer(params.inputWidth);

    let lastLayer: SimpleLayer = input;
    for (let i = 0; i < params.hiddenWidth.length; i++) {
      const hiddenWidth = params.hiddenWidth[i];

      const denseLinear = new DenseLinearLayer({
        prevLayer: lastLayer,
        size: params.hiddenWidth[i],
        initialWeight: Matrix.fromRaw(
          Array.from({ length: hiddenWidth }).map(() =>
            Array.from({ length: lastLayer.outputSize }).map(() =>
              params.parameterInitializer()
            )
          )
        ),
        useBias: true,
      });

      const activation = new ActivationLayer(
        denseLinear,
        params.hiddenActivationFuncs[i % params.hiddenActivationFuncs.length]
      );

      if (params.normalizationLayer) {
        lastLayer = new NormLayer(activation);
      } else {
        lastLayer = activation;
      }
    }

    const denseLinearOutput = new DenseLinearLayer({
      prevLayer: lastLayer,
      size: params.outputWidth,
      initialWeight: Matrix.fromRaw(
        Array.from({ length: params.outputWidth }).map(() =>
          Array.from({ length: lastLayer.outputSize }).map(() =>
            params.parameterInitializer()
          )
        )
      ),
      useBias: true,
    });

    let output: SimpleLayer;

    if ((params.outputActivationFunc as any).type === "softmax") {
      output = new SoftmaxLayer(
        denseLinearOutput,
        (params.outputActivationFunc as any).temperature,
        'final'
      );
    } else {
      output = new ActivationLayer(
        denseLinearOutput,
        params.outputActivationFunc as ActivationFunc
      );
    }

    super(
      new MatrixLayer({ rows: [input] }),
      new MatrixLayer({ rows: [output] })
    );
  }
}
