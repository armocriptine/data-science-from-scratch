import { ActivationFunc } from "../functions/Function";
import { Node } from "./Node";

export class ActivationNode extends Node<number> {
  constructor(
    public readonly incomingNode: Node,
    public readonly activationFunc: ActivationFunc<number>
  ) {
    super([incomingNode], activationFunc);
  }

  protected override _preactivate(sessionId: string, training: boolean): number {
    return this.incomingNode.activate(sessionId, training);
  }
  
  protected _getDerivativeByWrt(derivative: number): number {
    return derivative;
  }
}
