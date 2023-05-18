import { Identity } from "../functions/ActivationFunctions";
import { Node } from "./Node";

export class ParameterNode extends Node<number> {
  public accumulatedGradient: number = 0;
  public additionalData: any = null;

  constructor(public value: number, public learnable = true) {
    super([], new Identity());
  }

  public override prebackprop(sessionId: string, training: boolean): number {
    const prebackprop = super.prebackprop(sessionId, training);
  
    this.accumulatedGradient += prebackprop;

    return prebackprop;
  }

  public adjustValue(amount: number): void {
    this.value += amount;
    this.accumulatedGradient = 0;
  }

  protected _preactivate() {
    return this.value;
  }

  protected _getDerivativeByWrt(): number {
    throw new Error("Method not implemented.");
  }

  public override getAllParametersInPath(): ParameterNode[] {
    if (this.traversedByGetAllParameters) return [];
    this.traversedByGetAllParameters = true;
    return [this];
  }
}
