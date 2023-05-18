import { ActivationFunc } from "../functions/Function";
import { ActivationNode } from "../nodes/Activation";
import { SimpleLayer } from "./Layer";

export class ActivationLayer extends SimpleLayer {
  constructor(
    public readonly incomingLayer: SimpleLayer,
    public readonly activationFunc: ActivationFunc
  ) {
    const incomingNodes = incomingLayer.outputNodes;

    const innerNodes = Array.from({ length: incomingNodes.length }).map(
      (_, i) => new ActivationNode(incomingNodes[i], activationFunc)
    );

    super(innerNodes)
  }
}
