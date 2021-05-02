using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.Rendering;

// Shapes © Freya Holmér - https://twitter.com/FreyaHolmer/
// Website & Documentation - https://acegikmo.com/shapes/
namespace Shapes {

	internal struct DrawStyle {

		public static DrawStyle @default = new DrawStyle {
			color = Color.white,
			renderState = new RenderState {
				zTest = ShapeRenderer.DEFAULT_ZTEST,
				zOffsetFactor = ShapeRenderer.DEFAULT_ZOFS_FACTOR,
				zOffsetUnits = ShapeRenderer.DEFAULT_ZOFS_UNITS,
				stencilComp = ShapeRenderer.DEFAULT_STENCIL_COMP,
				stencilOpPass = ShapeRenderer.DEFAULT_STENCIL_OP,
				stencilRefID = ShapeRenderer.DEFAULT_STENCIL_REF_ID,
				stencilReadMask = ShapeRenderer.DEFAULT_STENCIL_MASK,
				stencilWriteMask = ShapeRenderer.DEFAULT_STENCIL_MASK
			},
			blendMode = ShapesBlendMode.Transparent,
			scaleMode = ScaleMode.Uniform,
			detailLevel = DetailLevel.Medium,
			lineThickness = 0.05f,
			lineThicknessSpace = ThicknessSpace.Meters,
			lineDashStyle = DashStyle.DefaultDashStyleLine,
			lineEndCaps = LineEndCap.Round,
			lineGeometry = LineGeometry.Billboard,
			polygonTriangulation = PolygonTriangulation.EarClipping,
			polygonShapeFill = new ShapeFill(),
			polylineGeometry = PolylineGeometry.Billboard,
			polylineJoins = PolylineJoins.Round,

			// disc
			discGeometry = DiscGeometry.Flat2D,
			discRadius = 1f,
			ringThickness = 0.05f,
			ringThicknessSpace = ThicknessSpace.Meters,
			discRadiusSpace = ThicknessSpace.Meters,
			ringDashStyle = DashStyle.DefaultDashStyleRing,

			// regular polygon
			regularPolygonRadius = 1f,
			regularPolygonSideCount = 6,
			regularPolygonGeometry = RegularPolygonGeometry.Flat2D,
			regularPolygonThickness = 0.05f,
			regularPolygonThicknessSpace = ThicknessSpace.Meters,
			regularPolygonRadiusSpace = ThicknessSpace.Meters,
			regularPolygonShapeFill = new ShapeFill(),

			sphereRadius = 1f,
			sphereRadiusSpace = ThicknessSpace.Meters,
			cuboidSizeSpace = ThicknessSpace.Meters,
			torusThicknessSpace = ThicknessSpace.Meters,
			torusRadiusSpace = ThicknessSpace.Meters,
			coneSizeSpace = ThicknessSpace.Meters,
			font = ShapesAssets.Instance.defaultFont,
			fontSize = 1f,
			textAlign = TextAlign.Center
		};

		// globally shared render state styles
		public RenderState renderState;
		public Color color;
		public ShapesBlendMode blendMode; // technically a render state rather than property, but we swap shaders here instead
		public ScaleMode scaleMode;
		public DetailLevel detailLevel;

		// shared line & polyline states
		public float lineThickness;
		public ThicknessSpace lineThicknessSpace;

		// line states
		public LineEndCap lineEndCaps;
		public LineGeometry lineGeometry;

		// polygon states
		public PolygonTriangulation polygonTriangulation;
		public ShapeFill polygonShapeFill;

		// line dashes
		public DashStyle lineDashStyle;
		public DashStyle ringDashStyle;

		// polyline states
		public PolylineGeometry polylineGeometry;
		public PolylineJoins polylineJoins;

		// disc & ring states
		public float discRadius;
		public DiscGeometry discGeometry;
		public float ringThickness;
		public ThicknessSpace ringThicknessSpace;
		public ThicknessSpace discRadiusSpace;

		// regular polygon states
		public float regularPolygonRadius;
		public int regularPolygonSideCount;
		public RegularPolygonGeometry regularPolygonGeometry;
		public float regularPolygonThickness;
		public ThicknessSpace regularPolygonThicknessSpace;
		public ThicknessSpace regularPolygonRadiusSpace;
		public ShapeFill regularPolygonShapeFill;

		// 3D shape states
		public float sphereRadius;
		public ThicknessSpace sphereRadiusSpace;
		public ThicknessSpace cuboidSizeSpace;
		public ThicknessSpace torusThicknessSpace;
		public ThicknessSpace torusRadiusSpace;
		public ThicknessSpace coneSizeSpace;

		// text states
		public TMP_FontAsset font;
		public float fontSize;
		public TextAlign textAlign;
	}

}