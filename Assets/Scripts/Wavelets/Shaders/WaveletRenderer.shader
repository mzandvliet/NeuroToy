Shader "Unlit/WaveletRenderer"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Gain("Gain", Float) = 1.0
        _Bias("Bias", Float) = 0.0
        _HueShift("Hue Shift", Float) = 0.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float _Bias = 0;
            float _Gain = 1;
            float _HueShift = 0;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            float3 HUEtoRGB(in float H)
            {
                float R = abs(H * 6 - 3) - 1;
                float G = 2 - abs(H * 6 - 2);
                float B = 2 - abs(H * 6 - 4);
                return saturate(float3(R,G,B));
            }

            float3 hsvToRgb(float3 HSV)
            {
                float3 RGB = HUEtoRGB(HSV.x);
                return ((RGB - 1) * HSV.y + 1) * HSV.z;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // sample the data
                float v = tex2D(_MainTex, i.uv).x;

                /* 
                
                First we try to undo the visual skew towards the lower frequencies by
                diminishing the energy shown there proportionally, so that the image
                can give a decent estimate of energy per area.

                Relates to this line in the C# code:

                float Scale2Freq() {
                    return cfg.lowestScale + (Mathf.Pow(cfg.scalePowBase, scale) - 1f) * cfg._scaleNormalizationFactor;
                }

                */
                
                v *= log(1 + i.uv.y) / log(1.0251);

                v += _Bias;
                v *= _Gain;
                
                /* 
                Logarithmic scaling which emphasizes higher frequency action
                */
                v = sign(v) * log10(1.0 + abs(v));

                // Simple 2 color spectrum
                // float4 colPos = float4(0, 100.0 / 255.0, 255.0 / 255.0,1);
                // float4 colNeg = float4(255.0 / 255.0,100.0 / 255.0,0,1);
                // float4 col =
                //     colNeg * max(0, -1.0 * v) +
                //     colPos * max(0,  1 * v)


                /*
                Hue shifts based on magnitude, so we get contours
                */
                float hue = frac(_HueShift + 0.5 + 0.5 * v);
                float4 col = float4(hsvToRgb(float3(hue, 1, abs(v))), 1);

                // apply fog
                UNITY_APPLY_FOG(i.fogCoord, col);
                return col;
            }
            ENDCG
        }
    }
}
