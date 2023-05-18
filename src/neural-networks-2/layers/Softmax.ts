import { SoftmaxNode } from "../nodes/Softmax";
import { SimpleLayer } from "./Layer";

export class SoftmaxLayer extends SimpleLayer {
  constructor(prevLayer: SimpleLayer, temperature: number, name: string) {
    const softmaxNodes = prevLayer.outputNodes.map(
      (n, i) =>
        new SoftmaxNode(
          n,
          prevLayer.outputNodes.filter((_, j) => i !== j),
          temperature,
          name
        )
    );

    super(softmaxNodes);
  }
}
