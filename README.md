# SubstanceShaderUE
Substance Designer shader for 1-1 parity with UE4 viewport

Installation steps:
1. Locate your Substance Designer install directory.
2. Locate the resources/view3d/shaders folder inside your Substance Designer install directory
    \\Substance Designer\resources\view3d\shaders
3. Copy contents of this repository to the resources/view3d/shaders folder.
   You should end up with UE4PBR.glslfx in that folder, and a new folder inside called UE4Shading containg 2 files.
4. Reload shaders in Substance Designer
   3D VIEW>Materials>Default>Definitions>Reload All Shaders
5. Set 3D VIEW shader to UE4PBR.
   3D VIEW>Materials>Default>Definitions>UE4PBR>Parallax Occlusion/Tessellation
   
(optional) Set shader as your default 3D VIEW shader.
   Edit>Preferences>PROJECTS>Project>3D View>Default shader <- point this to UE4PBR.glslfx in that folder.

   

   
Cloth BxDF - Charlie distribution term & Ashkimin visibility term

*It is recommended to enable the directional light to correctly view the falloff

1. Cloth (mask usage)				 - Blends between default lit microfacet BRDF & cloth microfuzz BRDF
2. Fuzz Colour (transmissive usage)	 - Reflectance of fibres