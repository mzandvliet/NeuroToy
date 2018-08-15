using UnityEngine;

/* 
    Let's make some nodes. 

    they're all some f(a, b) thing with a forward
    calculation path, and given a gradient can
    route it back through, with respect to any
    of the inputs.

    We build a graph of these nodes, then execute
    operations on them.
*/

namespace BackPropPractice {
    public class BackPropagation : MonoBehaviour {

        private void Awake() {
            var a = new ConstNode(5f);
            var b = new ConstNode(3f);
            var add = new AddNode(a, b);

            const float rate = 0.1f;

            for (int i = 0; i < 100; i++) {
                float result = add.Forward();
                float target = 10f;
                float dCdO = target - result; // Note: get from SumSquareLoss node

                float dA = add.BackwardA(dCdO);
                float dB = add.BackwardB(dCdO);

                Debug.Log(a.Value + " + " + b.Value + " = " + result);

                a.Value += dA * rate;
                b.Value += dB * rate;
            }
        }
    }

    public interface IFloatNode {
        float Forward();
    }

    public class ConstNode : IFloatNode {
        public float Value {
            get;
            set;
        }

        public ConstNode(float value) {
            Value = value;
        }

        public float Forward() {
            return Value;
        }
    }

    public class AddNode : IFloatNode {
        public IFloatNode A {
            get;
            private set;
        }

        public IFloatNode B {
            get;
            private set;
        }

        public AddNode(IFloatNode a, IFloatNode b) {
            A = a;
            B = b;
        }

        public float Forward() {
            return A.Forward() + B.Forward();
        }

        public float BackwardA(float error) {
            return error;
        }

        public float BackwardB(float error) {
            return error;
        }
    }

    // public class SquareLossNode {
    //     public FloatNode A {
    //         get;
    //         private set;
    //     }

    //     public FloatNode B {
    //         get;
    //         private set;
    //     }

    //     public SquareLossNode(FloatNode a, FloatNode b) {
    //         A = a;
    //         B = b;
    //     }

    //     public float Forward() {
    //         return 0.5f * (A.Value - B.Value) * (A.Value - B.Value);
    //     }

    //     public float Backward() {
    //         return A.Value - B.Value;
    //     }
    // }
}
