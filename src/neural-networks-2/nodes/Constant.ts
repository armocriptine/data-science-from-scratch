import { Identity } from "../functions/ActivationFunctions";
import { Node } from "./Node";

export class ConstantNode extends Node {
  constructor(public readonly value: number) {
    super([], new Identity());
  }

  protected _preactivate() {
    return this.value;
  }

  protected _getDerivativeByWrt(): number {
    return 0;
  }
}
