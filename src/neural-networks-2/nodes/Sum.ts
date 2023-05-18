import { Sum } from "../functions/ActivationFunctions";
import { Node } from "./Node";

export class SumNode extends Node<number[]> {
  constructor(public readonly incomingNodes: Node[]) {
    super(incomingNodes, new Sum());
  }

  protected override _preactivate(sessionId: string, training: boolean): number[] {
    const inputs = this.incomingNodes.map((n) => n.activate(sessionId, training));

    return inputs as number[];
  }

  protected _getDerivativeByWrt(): number {
    return 1;
  }
}
