import { Identity } from "../functions/ActivationFunctions";
import { Node } from "./Node";

export class InputNode extends Node {
  public input: number | null = null;

  constructor() {
    super([], new Identity());
  }

  protected _preactivate(): number {
    if (this.input == null) throw new Error("Input is not set!");

    return this.input;
  }

  protected _getDerivativeByWrt(): number {
    throw new Error("Method not implemented.");
  }
}
