import { Normalize, NormalizeInput } from "../functions/ActivationFunctions";
import { Node } from "./Node";

export class NormNode extends Node<NormalizeInput> {
  constructor(
    public readonly mainNode: Node,
    public readonly otherNodes: Node[],
    public readonly e: number
  ) {
    super([mainNode, ...otherNodes], new Normalize(e));
  }

  protected _preactivate(sessionId: string, training: boolean): NormalizeInput {
    const main = this.mainNode.activate(sessionId, training);
    const others = this.otherNodes.map((n) => n.activate(sessionId, training));

    return { main, others };
  }

  protected _getDerivativeByWrt(
    derivatives: NormalizeInput,
    wrt: Node
  ): number {
    if (wrt === this.mainNode) {
      return derivatives.main;
    } else {
      const index = this.otherNodes.indexOf(wrt);
      return derivatives.others[index];
    }
  }
}
