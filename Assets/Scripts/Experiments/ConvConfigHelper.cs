using UnityEngine;
using UnityEditor;

/*
 Todo: one validation function, used in gui and in code
 */

[System.Serializable]
public class ConvConfig {
    [SerializeField] public int InputWidth;
    [SerializeField] public ConvLayerConfig[] Layers;
}

[CustomPropertyDrawer(typeof(ConvConfig))]
public class ConvConfigDrawer : PropertyDrawer {
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label) {
        EditorGUI.BeginProperty(position, label, property);
        SerializedProperty list = property.FindPropertyRelative("Layers");

        var inWidthProp = property.FindPropertyRelative("InputWidth");
        var inWidthRect = new Rect(position.x, position.y, position.width, 16);
        EditorGUI.PropertyField(inWidthRect, inWidthProp, new GUIContent("Input Width"));

        position.y += 16;

        Color col = GUI.color;
        int lastOutWidth = inWidthProp.intValue;

        for (int i = 0; i < list.arraySize; i++) {
            var prop = list.GetArrayElementAtIndex(i);
            int kernWidth = prop.FindPropertyRelative("KernWidth").intValue;
            int stride = prop.FindPropertyRelative("Stride").intValue;
            int padding = prop.FindPropertyRelative("Padding").intValue;
            int outWidth = NNBurst.ConvLayer2D.GetOutputWidth(lastOutWidth, kernWidth, stride, padding);
            GUI.color = outWidth == -1 ? Color.red : col;
            
            var propRect = new Rect(position.x, position.y + i * ConvLayerConfigDrawer.PropHeight, position.width, position.height);
            EditorGUI.PropertyField(propRect, prop, new GUIContent("Layer "+i));

            var delRect = new Rect(position.x + 80, position.y + i * ConvLayerConfigDrawer.PropHeight, 20, 15);
            if (GUI.Button(delRect, "-")) {
                list.DeleteArrayElementAtIndex(i);
            }

            lastOutWidth = outWidth;
        }

        GUI.color = col;

        var addRect = new Rect(position.x + position.width / 2f, position.y + list.arraySize * ConvLayerConfigDrawer.PropHeight, 20, 16);
        if (GUI.Button(addRect, "+")) {
            list.InsertArrayElementAtIndex(list.arraySize);
        }

        var outSizeRect = new Rect(position.x, position.y + list.arraySize * ConvLayerConfigDrawer.PropHeight, position.width, 16);
        EditorGUI.LabelField(outSizeRect, new GUIContent("Output width: " + lastOutWidth));

        EditorGUI.EndProperty();
    }

    public override float GetPropertyHeight(SerializedProperty property, GUIContent label) {
        SerializedProperty list = property.FindPropertyRelative("Layers");
        return ConvLayerConfigDrawer.PropHeight * list.arraySize + 2*16;
    }
}

[System.Serializable]
public struct ConvLayerConfig {
    [SerializeField] public int KernWidth;
    [SerializeField] public int Stride;
    [SerializeField] public int Padding;
}

[CustomPropertyDrawer(typeof(ConvLayerConfig))]
public class ConvLayerConfigDrawer : PropertyDrawer {
    // Draw the property inside the given rect
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label) {
        // Using BeginProperty / EndProperty on the parent property means that
        // prefab override logic works on the entire property.
        EditorGUI.BeginProperty(position, label, property);

        // Draw label
        position = EditorGUI.PrefixLabel(position, GUIUtility.GetControlID(FocusType.Passive), label);

        // Don't make child fields be indented
        var indent = EditorGUI.indentLevel;
        EditorGUI.indentLevel = 0;

        // Calculate rects
        var widthRect = new Rect(position.x, position.y, position.width, 16);
        var strideRect = new Rect(position.x, position.y + 20, position.width, 16);
        var paddingRect = new Rect(position.x, position.y + 40, position.width, 16);

        var kernWidth = property.FindPropertyRelative("KernWidth");
        var stride = property.FindPropertyRelative("Stride");
        var padding = property.FindPropertyRelative("Padding");

        // Draw fields - passs GUIContent.none to each so they are drawn without labels
        EditorGUI.PropertyField(widthRect, kernWidth, new GUIContent("Width"));
        EditorGUI.PropertyField(strideRect, stride, new GUIContent("Stride"));
        EditorGUI.PropertyField(paddingRect, padding, new GUIContent("Padding"));

        // Set indent back to what it was
        EditorGUI.indentLevel = indent;

        EditorGUI.EndProperty();
    }

    public override float GetPropertyHeight(SerializedProperty property, GUIContent label) {
        return PropHeight;
    }

    public static readonly float PropHeight = 60f;
}

public class ConvConfigHelper : MonoBehaviour {
    [SerializeField] private ConvConfig _config;
    
    private void Awake() {
        bool valid = ValidateConfig(_config, _config.InputWidth);
        Debug.Log("Config valid? " + valid);
    }
    
    private static bool ValidateConfig(ConvConfig conf, int inWidth) {
        int lastOutWidth = inWidth;

        for (int i = 0; i < conf.Layers.Length; i++) {
            int outputSize = NNBurst.ConvLayer2D.GetOutputWidth(
                lastOutWidth,
                conf.Layers[i].KernWidth, 
                conf.Layers[i].Stride, 
                conf.Layers[i].Padding);
            
            if (outputSize == -1) {
                return false;
            }
        }

        return true;
    }
}