import { UniformDistribution } from "../../prob/distributions/Uniform";
import { Vector } from "../../vector/Vector";
import { ActivationFunc } from "../functions/Function";
import { ActivationLayer } from "./Activation";
import { MatAddLayer } from "./MatAdd";
import { MatMulLayer } from "./MatMul";
import { MatrixLayer } from "./Matrix";
import { ParameterLayer } from "./Parameter";

export class TransformerFeedforwardLayer extends MatrixLayer {
  constructor(params: {
    input: MatrixLayer;
    size: number;
    activationFunc: ActivationFunc;
  }) {
    const ffPreactWeightBound = Math.sqrt(
      6 / (params.input.width + params.size)
    );
    const ffPreactWeightInit = new UniformDistribution(
      -ffPreactWeightBound,
      ffPreactWeightBound
    );
    const feedforwardPreactWeight = new MatrixLayer({
      rows: Array.from({ length: params.input.width }).map(
        () =>
          new ParameterLayer(
            params.size,
            Vector.from(ffPreactWeightInit, params.size)
          )
      ),
    });

    const feedforwardPreactLinear = new MatMulLayer(
      params.input,
      feedforwardPreactWeight
    );

    const feedforwardPreactBias = new MatrixLayer({
      rows: Array.from({ length: feedforwardPreactLinear.height }).map(
        () =>
          new ParameterLayer(
            feedforwardPreactLinear.width,
            Vector.zero(feedforwardPreactLinear.width)
          )
      ),
    });

    const feedforwardBiasedPreact = new MatAddLayer(
      feedforwardPreactLinear,
      feedforwardPreactBias
    );

    const feedforwardActivation = new MatrixLayer({
      rows: feedforwardBiasedPreact.rows.map(
        (row) => new ActivationLayer(row, params.activationFunc)
      ),
    });

    const attProjWeightBound = Math.sqrt(
      6 / (feedforwardActivation.width + params.input.width)
    );
    const attProjWeightInit = new UniformDistribution(
      -attProjWeightBound,
      attProjWeightBound
    );
    const feedforwardPostactWeight = new MatrixLayer({
      rows: Array.from({ length: feedforwardActivation.width }).map(
        () =>
          new ParameterLayer(
            params.input.width,
            Vector.from(attProjWeightInit, params.input.width)
          )
      ),
    });

    const feedforwardPostactLinear = new MatMulLayer(
      feedforwardActivation,
      feedforwardPostactWeight
    );

    const feedforwardPostactBias = new MatrixLayer({
      rows: Array.from({ length: feedforwardPostactLinear.height }).map(
        () =>
          new ParameterLayer(
            feedforwardPostactLinear.width,
            Vector.zero(feedforwardPostactLinear.width)
          )
      ),
    });

    const feedforwardBiasedPostact = new MatAddLayer(
      feedforwardPostactLinear,
      feedforwardPostactBias
    );

    super({ rows: feedforwardBiasedPostact.rows });
  }
}
