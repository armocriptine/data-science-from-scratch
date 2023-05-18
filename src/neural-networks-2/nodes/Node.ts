import { sum } from "../../utils/helpers";
import { ActivationFunc } from "../functions/Function";
import { ParameterNode } from "./Parameter";

export abstract class Node<T = any> {
  private lastPostactivation: number | null = null;
  private lastPreactivation: T | null = null;

  private lastPrebackprop: number | null = null;

  private lastSessionId: string | null = null;

  protected traversedByGetAllParameters: boolean = false;

  constructor(
    public readonly incomingNodes: Node[],
    public readonly activationFunction: ActivationFunc<T>
  ) {
    for (const incomingNode of incomingNodes) {
      if (!incomingNode) {
        console.log("!");
      }
      incomingNode.outgoingNodes.push(this);
    }
  }

  public outgoingNodes: Node<any>[] = [];

  public activate(sessionId: string, training: boolean): number {
    if (this.lastSessionId !== sessionId) {
      this.clearCache();
    }

    if (this.lastPostactivation !== null) {
      return this.lastPostactivation;
    }

    const preactivation = this.preactivate(sessionId, training);
    if (Number.isNaN(preactivation)) {
      console.log('!')
    }
    const postactivation = this._activate(preactivation);

    this.lastPreactivation = preactivation;
    this.lastPostactivation = postactivation;
    this.lastSessionId = sessionId;

    return postactivation;
  }

  private preactivate(sessionId: string, training: boolean): T {
    if (this.lastPreactivation == null) {
      this.lastPreactivation = this._preactivate(sessionId, training);
    }

    return this.lastPreactivation;
  }

  protected abstract _preactivate(sessionId: string, training: boolean): T;

  protected _activate(preactivation: T): number {
    return this.activationFunction.evaluate(preactivation);
  }

  public setLossGradient(gradient: number): void {
    this.lastPrebackprop = gradient;
  }

  public prebackprop(sessionId: string, training: boolean): number {
    if (this.lastPrebackprop === null) {
      const prebackprops = this.outgoingNodes.map((n) => n.backprop(sessionId, training, this));
      this.lastPrebackprop = prebackprops.reduce(sum, 0);
    }

    return this.lastPrebackprop;
  }

  public backprop(sessionId: string, training: boolean, wrt: Node<any>): number {
    const prebackprop = this.prebackprop(sessionId, training);
    return Math.min(10, Math.max(-10, this.differentiate(sessionId, training, wrt) * prebackprop)); // gradient clipping
  }

  public differentiate(sessionId: string, training: boolean, wrt: Node<any>): number {
    return this._getDerivativeByWrt(
      this.activationFunction.differentiate(this.preactivate(sessionId, training)),
      wrt
    );
  }

  protected abstract _getDerivativeByWrt(
    derivatives: T,
    wrt: Node<any>
  ): number;

  private clearCache(): void {
    this.lastPostactivation = null;
    this.lastPreactivation = null;
    this.lastPrebackprop = null;
  }

  public getAllParametersInPath(): ParameterNode[] {
    if (this.traversedByGetAllParameters) return [];

    this.traversedByGetAllParameters = true;

    return this.incomingNodes.flatMap((n) => n.getAllParametersInPath());
  }
}
