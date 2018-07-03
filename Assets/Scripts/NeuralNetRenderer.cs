using System;
using UnityEngine;


/* Todo:
 * 
 * No reason for this to be a monobehaviour
 * Add labels for inputs and outputs, possibly grouping ins/outs that belong
 * to the same data structure.
 */
public class NeuralNetRenderer : MonoBehaviour {
    [SerializeField] private Texture2D _neuronTex;

    private Network _net;

    public void SetTarget(Network network) {
        _net = network;
    }

    private void OnGUI() {
        if (Event.current.type != EventType.Repaint || _net == null) {
            return;
        }

        Vector2 windowTopRight = new Vector2(100f, 50f);
        Vector2 windowSize = new Vector2(Screen.width * 0.9f, Screen.height * 1.0f);

        int biggestLayerSize = 0;
        for (int i = 0; i < _net.Layers.Count; i++) {
            if (_net.Layers[i].Count > biggestLayerSize) {
                biggestLayerSize = _net.Layers[i].Count;
            }
        }

        float vertStep = windowSize.y / _net.Layers.Count;

        for (int l = 0; l < _net.Layers.Count; l++) {
            for (int n = 0; n < _net.Layers[l].Count; n++) {
                // Draw a texture for each neuron in the layer
                float xStep = windowSize.x / _net.Layers[l].Count;
                Vector2 offsetCur = new Vector2(0f, l * vertStep);
                Vector2 curNPos =
                        windowTopRight +
                        offsetCur +
                        new Vector2(n * xStep, 0f);

                float act = _net.Layers[l].Outputs[n];
                float absAct = Mathf.Abs(act);
                GUI.color = new Color(absAct, absAct, absAct, 1f);
                GUI.DrawTexture(
                    new Rect(curNPos.x - 2f, curNPos.y - 2f, 4f, 4f),
                    _neuronTex);

                if (l < 1) {
                    continue;
                }

                // If we have previous layer, draw connections from those neurons to
                // The ones in the current layer
                float xStepPrev = windowSize.x / _net.Layers[l-1].Count;
                Vector2 offsetPrev = new Vector2(0f, (l-1) * vertStep);

                for (int w = 0; w < _net.Layers[l-1].Count; w++) {
                    Vector2 prevNPos = 
                        windowTopRight +
                        offsetPrev +
                        new Vector2(w * xStepPrev, 0f);
                    
                    float wVal = _net.Layers[l].Weights[n,w];
                    float synAct = _net.Layers[l - 1].Outputs[w] * wVal;

                    Color wCol = new Color(Mathf.Max(0f, -wVal), 0f, Mathf.Max(0f, wVal), Mathf.Abs(synAct));
                    float width = 0.05f + Mathf.Pow(synAct, 1f) * 0.5f;
                    Drawing.DrawLine(prevNPos, curNPos, wCol, width, true);
                }
            }
        }
    }
}