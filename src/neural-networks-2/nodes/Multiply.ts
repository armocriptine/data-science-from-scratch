import {
  Multiply,
  MultiplyFunctionInput,
} from "../functions/ActivationFunctions";
import { Node } from "./Node";

export class MultiplyNode extends Node<MultiplyFunctionInput> {
  constructor(public leftNode: Node, public rightNode: Node) {
    super([leftNode, rightNode], new Multiply());
  }

  protected _preactivate(sessionId: string, training: boolean): MultiplyFunctionInput {
    const left = this.leftNode.activate(sessionId, training);
    const right = this.rightNode.activate(sessionId, training);

    return { left, right };
  }

  protected _getDerivativeByWrt(
    derivative: MultiplyFunctionInput,
    wrt: Node
  ): number {
    if (wrt === this.leftNode) {
      return derivative.left;
    } else {
      return derivative.right;
    }
  }
}
