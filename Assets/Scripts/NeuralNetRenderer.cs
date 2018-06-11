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

    private FeedForwardNetwork _net;

    public void SetTarget(FeedForwardNetwork network) {
        _net = network;
    }

    private void OnGUI() {
        if (Event.current.type != EventType.Repaint || _net == null) {
            return;
        }

        Vector2 windowTopRight = new Vector2(100f, 50f);
        Vector2 windowSize = new Vector2(Screen.width * 0.9f, Screen.height * 1.0f);

        int biggestLayerSize = 1;
        for (int i = 0; i < _net.Topology.Length; i++) {
            if (_net.Topology[i] > biggestLayerSize) {
                biggestLayerSize = _net.Topology[i];
            }
        }

        float vertStep = windowSize.y / _net.Topology.Length;
        float horStep = windowSize.x / biggestLayerSize;

        for (int l = 0; l < _net.Topology.Length; l++) {
            for (int n = 0; n < _net.Topology[l]; n++) {
                // Draw a texture for each neuron in the layer
                int horOffsetCur = (biggestLayerSize - _net.Topology[l]) / 2;
                Vector2 offsetCur = new Vector2(horOffsetCur * horStep, 0f);

                Vector2 curNPos =
                        windowTopRight +
                        offsetCur +
                        new Vector2(n * horStep, l * vertStep);

                float act = _net.A[l][n];
                //GUI.color = new Color(Mathf.Max(0f, -act), 0f, Mathf.Max(0f, act), 1f);
                float absAct = Mathf.Abs(act);
                GUI.color = new Color(absAct, absAct, absAct, 1f);
                GUI.DrawTexture(
                    new Rect(curNPos.x - 8f, curNPos.y - 8f, 16f, 16f),
                    _neuronTex);

                if (l < 1) {
                    continue;
                }

                // If we have previous layer, draw connections from those neurons to
                // The ones in the current layer
                int hOffsetPrev = (biggestLayerSize - _net.Topology[l - 1]) / 2;
                Vector2 offsetPrev = new Vector2(hOffsetPrev * horStep, 0f);

                for (int w = 0; w < _net.Topology[l - 1]; w++) {
                    Vector2 prevNPos = 
                        windowTopRight +
                        offsetPrev +
                        new Vector2(w * horStep, (l-1) * vertStep);
                    

                    float wVal = _net.W[l][n][w];

                    float synAct = _net.A[l - 1][w] * wVal;

                    Color wCol = new Color(Mathf.Max(0f, -wVal), 0f, Mathf.Max(0f, wVal), Mathf.Abs(synAct));
                    float width = 0.1f + Mathf.Pow(synAct, 2f) * 1.1f;
                    Drawing.DrawLine(prevNPos, curNPos, wCol, width, true);
                }
            }
        }
    }
}