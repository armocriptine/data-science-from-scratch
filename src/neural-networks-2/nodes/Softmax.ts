import { Softmax, SoftmaxInput } from "../functions/ActivationFunctions";
import { Node } from "./Node";

export class SoftmaxNode extends Node<SoftmaxInput> {
  constructor(
    public readonly numeratorNode: Node,
    public readonly otherNodes: Node[],
    public readonly temperature: number,
    public readonly name: string,
  ) {
    super([numeratorNode, ...otherNodes], new Softmax());
  }

  protected _preactivate(sessionId: string, training: boolean): SoftmaxInput {
    const numerator = this.numeratorNode.activate(sessionId, training);
    const others = this.otherNodes.map((n) => n.activate(sessionId, training));

    return {
      numerator: numerator / this.temperature,
      others: others.map((x) => x / this.temperature),
    };
  }

  protected _getDerivativeByWrt(derivatives: SoftmaxInput, wrt: Node): number {
    if (wrt === this.numeratorNode) {
      return derivatives.numerator;
    } else {
      const index = this.otherNodes.indexOf(wrt);
      return derivatives.others[index];
    }
  }
}
