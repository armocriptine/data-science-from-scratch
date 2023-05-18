import { sum } from "./helpers";

export class MovingAverage {
  private entries: number[] = [];
  private pointer = 0;

  public push(x: number) {
    if (this.entries.length < 100) {
      this.entries.push(x);
    } else {
      this.entries[this.pointer] = x;
      if (this.pointer >= 100) {
        this.pointer = 0;
      } else {
        this.pointer++;
      }
    }
  }

  public get average(): number {
    return this.entries.reduce(sum, 0) / this.entries.length;
  }
  public get count(): number {
    return this.entries.length;
  }
}
