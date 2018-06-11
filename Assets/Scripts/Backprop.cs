
namespace Backprop {
    public class Backprop {
        public void Run() {
            var def = new NetDefinition(2, 
                new LayerDefinition(2, LayerType.Deterministic, ActivationType.Sigmoid), 
                new LayerDefinition(1, LayerType.Deterministic, ActivationType.Sigmoid));
            var net = NetBuilder.Build(def);

            /* Todo:
             * - Generate batches input/output examples
             * - for each batch
             *      - for each example
             *              - forward pass
             *              - error gradient with respect to output layer
             *              - backpropagate
             *              - store resulting W and B nudges
             *      - average nudges, scale by learning rate, apply to net
             *      
             *      
             * First, just make sure you have a training set, have a working
             * forward pass, and can calculate gradient(C) = A(L) - Y, and
             * plot a graph for it, so we can see that go down, later.
             */

            NetUtils.Forward(net);
        }
    }
}
