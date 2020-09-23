Shader "Unlit/WaveletRenderer"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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
            float _playTime; // normalized play time within visible window
            float _bias;
            float _gain;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // sample the texture
                fixed4 col = tex2D(_MainTex, i.uv);

                col = (col - _bias) * _gain;

                // Todo: normalize, log scaling
                // float mag = math.log10(1f + tex[i].x);

                const float playHeadBand = 0.05;
                const float playHeadPow = 2.0;
                float playHeadProximity = pow(1.0 - clamp(abs(i.uv.x-_playTime), 0.0, playHeadBand) * (1.0 / playHeadBand), playHeadPow);
                // col = lerp(col, fixed4(1,1,1,1), playHeadProximity);
                col *= 1.5 + playHeadProximity * 8;

                // apply fog
                UNITY_APPLY_FOG(i.fogCoord, col);
                return col;
            }
            ENDCG
        }
    }
}
