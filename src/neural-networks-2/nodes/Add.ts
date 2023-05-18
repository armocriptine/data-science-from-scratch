import { Add } from "../functions/ActivationFunctions";
import { Node } from "./Node";

export class AddNode extends Node<{ a: number; b: number }> {
  constructor(
    private readonly leftNode: Node,
    public readonly rightNode: Node
  ) {
    super([leftNode, rightNode], new Add());
  }

  protected _preactivate(sessionId: string, training: boolean) {
    const a = this.rightNode.activate(sessionId, training);
    const b = this.leftNode.activate(sessionId, training);

    return { a, b };
  }

  protected _getDerivativeByWrt(): number {
    return 1;
  }
}
