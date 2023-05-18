import { Matrix } from "../../matrix/Matrix";
import { Vector } from "../../vector/Vector";
import { SimpleLayer } from "./Layer";
import { MatAddLayer } from "./MatAdd";
import { MatMulLayer } from "./MatMul";
import { MatrixLayer } from "./Matrix";
import { ParameterLayer } from "./Parameter";

export class DenseLinearLayer extends SimpleLayer {
  constructor(params: {
    prevLayer: SimpleLayer;
    size: number;
    initialWeight: Matrix;
    useBias: boolean;
  }) {
    const initialWeightRows = params.initialWeight.rows;

    const weightLayer = new MatrixLayer({
      rows: Array.from({ length: params.size }).map(
        (_, i) =>
          new ParameterLayer(params.prevLayer.outputSize, initialWeightRows[i])
      ),
    });

    const matMulLayer = new MatMulLayer(
      weightLayer,
      new MatrixLayer({ rows: [params.prevLayer] })
    );

    let matAddLayer: MatAddLayer | null = null;
    if (params.useBias) {
      const biasLayer = new MatrixLayer({
        rows: Array.from({ length: matMulLayer.height }).map(
          () =>
            new ParameterLayer(
              matMulLayer.width,
              Vector.zero(matMulLayer.width)
            )
        ),
      });

      matAddLayer = new MatAddLayer(biasLayer, matMulLayer);
    }

    super((matAddLayer ?? matMulLayer).cols[0].outputNodes);
  }
}
