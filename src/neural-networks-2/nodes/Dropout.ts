import _ from "underscore";
import { Identity } from "../functions/ActivationFunctions";
import { Node } from "./Node";

export class DropoutNode extends Node {
  private droppedOut = false;

  constructor(public readonly incomingNode: Node, public readonly dropoutRate: number) {
    super([incomingNode], new Identity());
  }

  protected _preactivate(sessionId: string, training: boolean) {
    if (training && _.random(1) <= this.dropoutRate) {
      this.droppedOut = true;
      return 0;
    } else {
      this.droppedOut = false;
      return this.incomingNode.activate(sessionId, training);
    }
  }

  protected _getDerivativeByWrt(): number {
    if (this.droppedOut) {
      return 0;
    } else {
      return 1;
    }
  }
}
