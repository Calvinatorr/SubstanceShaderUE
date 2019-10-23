/*
Copyright (c) 2018, Allegorithmic. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of the Allegorithmic nor the
     names of its contributors may be used to endorse or promote products
     derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ALLEGORITHMIC BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//////////////////////////////// Fragment shader
#version 330

#include "../common/common.glsl"
#include "../common/uvtile.glsl"
#include "../common/parallax.glsl"

in vec3 iFS_Normal;
in vec2 iFS_UV;
in vec3 iFS_Tangent;
in vec3 iFS_Binormal;
in vec3 iFS_PointWS;

out vec4 ocolor0;

uniform vec3 Lamp0Pos = vec3(0.0,0.0,70.0);
uniform vec3 Lamp0Color = vec3(1.0,1.0,1.0);
uniform float Lamp0Intensity = 1.0;
uniform vec3 Lamp1Pos = vec3(70.0,0.0,0.0);
uniform vec3 Lamp1Color = vec3(0.198,0.198,0.198);
uniform float Lamp1Intensity = 1.0;

uniform float AmbiIntensity = 1.0;
uniform float EmissiveIntensity = 1.0;

uniform int parallax_mode = 0;

uniform float tiling = 1.0;
uniform vec3 uvwScale = vec3(1.0, 1.0, 1.0);
uniform bool uvwScaleEnabled = false;
uniform float envRotation = 0.0;
uniform float tessellationFactor = 4.0;
uniform float heightMapScale = 1.0;
uniform bool flipY = true;
uniform bool perFragBinormal = true;
uniform bool sRGBBaseColor = false;

uniform sampler2D heightMap;
uniform sampler2D normalMap;
uniform sampler2D baseColorMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;
uniform sampler2D emissiveMap;
uniform sampler2D specularLevel;
uniform sampler2D opacityMap;
uniform sampler2D environmentMap;

uniform mat4 viewInverseMatrix;

// Number of miplevels in the envmap
uniform float maxLod = 12.0;

// Actual number of samples in the table
uniform int nbSamples = 16;

// Irradiance spherical harmonics polynomial coefficients
// This is a color 2nd degree polynomial in (x,y,z), so it needs 10 coefficients
// for each color channel
uniform vec3 shCoefs[10];


// This must be included after the declaration of the uniform arrays since they
// can't be passed as functions parameters for performance reasons (on macs)
#include "../UE4Shading/pbr_ibl.glsl"

uniform bool UseIBL = true;
uniform bool UseACES = true;
uniform bool UseDirLight = true;
uniform vec3 LightDir = vec3(1.0);
uniform vec3 LightColour = vec3(1.0);
uniform float LightIntensity = 1.0;


float fit_roughness(float r)
{
// Fit roughness values to a more usable curve
	return r;
}

// ACES Tonemapping
vec3 tonemapSCurve(vec3 x)
{
  	float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e),0.0,1.0);
}

void main()
{
	vec3 normalWS = iFS_Normal;
	vec3 tangentWS = iFS_Tangent;
	vec3 binormalWS = perFragBinormal ?
		fixBinormal(normalWS,tangentWS,iFS_Binormal) : iFS_Binormal;

	vec3 cameraPosWS = viewInverseMatrix[3].xyz;
	vec3 pointToLight0DirWS = Lamp0Pos - iFS_PointWS;
	float pointToLight0Length = length(pointToLight0DirWS);
	pointToLight0DirWS *= 1.0 / pointToLight0Length;
	vec3 pointToLight1DirWS = Lamp1Pos - iFS_PointWS;
	float pointToLight1Length = length(Lamp1Pos - iFS_PointWS);
	pointToLight1DirWS *= 1.0 / pointToLight1Length;
	vec3 pointToCameraDirWS = normalize(cameraPosWS - iFS_PointWS);

	// ------------------------------------------
	// Parallax
	
	vec2 uv;

	
	vec2 uvScale = vec2(1.0, 1.0);
	if (uvwScaleEnabled)
		uvScale = uvwScale.xy;
		
	uv = parallax_mode == 1 ? iFS_UV*tiling*uvScale : updateUV(
		heightMap,
		pointToCameraDirWS,
		normalWS, tangentWS, binormalWS,
		heightMapScale,
		iFS_UV,
		uvScale,
		tiling);
		

	uv = uv / tiling / uvScale;		
	bool disableFragment = hasToDisableFragment(uv);
	uv = uv * tiling * uvScale;
	uv = getUDIMTileUV(uv);

	// ------------------------------------------
	// Add Normal from normalMap
	vec3 fixedNormalWS = normalWS;  // HACK for empty normal textures
	vec3 normalTS = get2DSample(normalMap, uv, disableFragment, cDefaultColor.mNormal).xyz;
	if(length(normalTS)>0.0001)
	{
		normalTS = fixNormalSample(normalTS,flipY);
		fixedNormalWS = normalize(
			normalTS.x*tangentWS +
			normalTS.y*binormalWS +
			normalTS.z*normalWS );
	}

	// ------------------------------------------
	// Compute material model (diffuse, specular & roughness)
	float dielectricSpec = get2DSample(specularLevel, uv, disableFragment, cDefaultColor.mSpecularLevel).r;
	dielectricSpec = 0.08 * dielectricSpec;
	vec3 dielectricColor = vec3(dielectricSpec, dielectricSpec, dielectricSpec);
	const float minRoughness=1e-4;
	// Convert the base color from sRGB to linear (we should have done this when
	// loading the texture but there is no way to specify which colorspace is
	// uѕed for a given texture in Designer yet)
	vec3 baseColor = get2DSample(baseColorMap, uv, disableFragment, cDefaultColor.mBaseColor).rgb;
	if (sRGBBaseColor)
		baseColor = srgb_to_linear(baseColor);



	float metallic = get2DSample(metallicMap, uv, disableFragment, cDefaultColor.mMetallic).r;
	float roughness = get2DSample(roughnessMap, uv, disableFragment, cDefaultColor.mRoughness).r;
	roughness = max(minRoughness, fit_roughness(roughness));

	vec3 diffColor = baseColor * (1.0 - metallic);
	vec3 specColor = mix(dielectricColor, baseColor, metallic);

	// ------------------------------------------
	// Compute point lights contributions
	vec3 contrib0 = pointLightContribution(
			fixedNormalWS,
			pointToLight0DirWS,
			pointToCameraDirWS,
			diffColor,
			specColor,
			roughness,
			Lamp0Color,
			Lamp0Intensity,
			pointToLight0Length );
	vec3 contrib1 = pointLightContribution(
			fixedNormalWS,
			pointToLight1DirWS,
			pointToCameraDirWS,
			diffColor,
			specColor,
			roughness,
			Lamp1Color,
			Lamp1Intensity,
			pointToLight1Length );
	vec3 dirLightContrib = dirLightContribution(
			fixedNormalWS,
			normalize(LightDir),
			pointToCameraDirWS,
			diffColor,
			specColor,
			roughness,
			LightColour,
			LightIntensity );

	// ------------------------------------------
	// Image based lighting contribution
	float ao = get2DSample(aoMap, uv, disableFragment, cDefaultColor.mAO).r;

	vec3 contribE = computeIBL(
		environmentMap, envRotation, maxLod,
		nbSamples,
		normalWS, fixedNormalWS, tangentWS, binormalWS,
		pointToCameraDirWS,
		diffColor, specColor, roughness,
		AmbiIntensity * ao);

	// ------------------------------------------
	//Emissive
	vec3 emissiveContrib = get2DSample(emissiveMap, uv, disableFragment, cDefaultColor.mEmissive).rgb;
	emissiveContrib = srgb_to_linear(emissiveContrib) * EmissiveIntensity;

	// ------------------------------------------
	vec3 finalColor = contrib0 + contrib1 + emissiveContrib;
	
	if ( UseIBL )
		finalColor += contribE;
	
	if ( UseDirLight )
		finalColor += dirLightContrib;
	
	if ( UseACES )
	{
		//finalColor = pow(finalColor, vec3(2.2));
		finalColor = tonemapSCurve(finalColor);
		//finalColor = pow(finalColor, vec3(1.0/2.2));
	}
	
	// Final Color
	// Convert the fragment color from linear to sRGB for display (we should
	// make the framebuffer use sRGB instead).
	float opacity = get2DSample(opacityMap, uv, disableFragment, cDefaultColor.mOpacity).r;
	ocolor0 = vec4(finalColor, opacity);
}

