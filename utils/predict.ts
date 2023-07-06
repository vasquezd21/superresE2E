// Language: typescript
// Path: react-next\utils\predict.ts
import { getImageTensorFromPath } from './imageHelper';
import { runSuperResModel } from './modelHelper';

export async function inferenceSuperRes(path: string): Promise<[any,number]> {
  // 1. Convert image to tensor
  const imageTensor = await getImageTensorFromPath(path);
  // 2. Run model
  const [predictions, inferenceTime] = await runSuperResModel(imageTensor);
  // 3. Return Image and the amount of time it took to inference.
  return [predictions, inferenceTime];
}
export { getImageTensorFromPath };

