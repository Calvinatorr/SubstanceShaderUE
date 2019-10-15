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

#define PI 3.14159265358979323846
#define M_INV_PI 0.31830988618379067153776752674503
#define M_INV_LOG2 1.4426950408889634073599246810019
#define M_GOLDEN_RATIO 0.618034

// Added by Calvin

uniform bool UseUE4Shading = true;

vec3 saturate( vec3 x )
{
	return clamp( x, 0, 1 );
}

// [Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"]
vec3 F_Schlick( vec3 SpecularColor, float VoH )
{
	//float Fc = Pow5( 1 - VoH );					// 1 sub, 3 mul
	//return Fc + (1 - Fc) * SpecularColor;		// 1 add, 3 mad
	float Fc = 1.0 - VoH;
	Fc = Fc * Fc * Fc * Fc * Fc; // Pow5
	
	// Anything less than 2% is physically impossible and is instead considered to be shadowing
	//return clamp( 50.0 * SpecularColor.g, 0, 1 ) * Fc + (1 - Fc) * SpecularColor;
	return saturate( 50.0 * SpecularColor ) * Fc + (1.0 - Fc) * SpecularColor;
	
}

// GGX / Trowbridge-Reitz
// [Walter et al. 2007, "Microfacet models for refraction through rough surfaces"]
float D_GGX( float a2, float NoH )
{
	float d = ( NoH * a2 - NoH ) * NoH + 1;	// 2 mad
	return a2 / ( PI*d*d );					// 4 mul, 1 rcp
}

// Smith term for GGX
// [Smith 1967, "Geometrical shadowing of a random rough surface"]
float Vis_Smith( float a2, float NoV, float NoL )
{
	float Vis_SmithV = NoV + sqrt( NoV * (NoV - NoV * a2) + a2 );
	float Vis_SmithL = NoL + sqrt( NoL * (NoL - NoL * a2) + a2 );
	return rcp( Vis_SmithV * Vis_SmithL );
}

// Appoximation of joint Smith term for GGX
// [Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"]
float Vis_SmithJointApprox( float a2, float NoV, float NoL )
{
	float a = sqrt(a2);
	float Vis_SmithV = NoL * ( NoV * ( 1 - a ) + a );
	float Vis_SmithL = NoV * ( NoL * ( 1 - a ) + a );
	return 0.5 * rcp( Vis_SmithV + Vis_SmithL );
}

// End here


float normal_distrib(
	float ndh,
	float Roughness)
{
// use GGX / Trowbridge-Reitz, same as Disney and Unreal 4
// cf http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf p3
	float alpha = Roughness * Roughness;
	float tmp = alpha / max(1e-8,(ndh*ndh*(alpha*alpha-1.0)+1.0));
	return tmp * tmp * M_INV_PI;
}

vec3 fresnel(
	float vdh,
	vec3 F0)
{
// Schlick with Spherical Gaussian approximation
// cf http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf p3
	float sphg = pow(2.0, (-5.55473*vdh - 6.98316) * vdh);
	return F0 + (vec3(1.0, 1.0, 1.0) - F0) * sphg;
}

float G1(
	float ndw, // w is either Ln or Vn
	float k)
{
// One generic factor of the geometry function divided by ndw
// NB : We should have k > 0
	return 1.0 / ( ndw*(1.0-k) + k );
}

float visibility(
	float ndl,
	float ndv,
	float Roughness)
{
// Schlick with Smith-like choice of k
// cf http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf p3
// visibility is a Cook-Torrance geometry function divided by (n.l)*(n.v)
	float k = Roughness * Roughness * 0.5;
	return G1(ndl,k)*G1(ndv,k);
}

vec3 microfacets_brdf(
	vec3 Nn,
	vec3 Ln,
	vec3 Vn,
	vec3 Ks,
	float Roughness)
{
	vec3 Hn = normalize(Vn + Ln);
	float vdh = max( 0.0, dot(Vn, Hn) );
	float ndh = max( 0.0, dot(Nn, Hn) );
	float ndl = max( 0.0, dot(Nn, Ln) );
	float ndv = max( 0.0, dot(Nn, Vn) );
	
	if ( UseUE4Shading )
	{
		float a = Roughness * Roughness;
		float a2 = a*a;
		return F_Schlick(Ks, vdh) * ( D_GGX( a2, ndh ) * Vis_Smith( Roughness * Roughness, ndv, ndl ) / 4.0 );
	}
	else
		return fresnel(vdh,Ks) * ( normal_distrib(ndh,Roughness) * visibility(ndl,ndv,Roughness) / 4.0 );
}

vec3 microfacets_contrib(
	float vdh,
	float ndh,
	float ndl,
	float ndv,
	vec3 Ks,
	float Roughness)
{
// This is the contribution when using importance sampling with the GGX based
// sample distribution. This means ct_contrib = ct_brdf / ggx_probability
	
	if ( UseUE4Shading )
		return fresnel(vdh,Ks) * (visibility(ndl,ndv,Roughness) * vdh * ndl / ndh );
	else
		return fresnel(vdh,Ks) * (visibility(ndl,ndv,Roughness) * vdh * ndl / ndh );
}

vec3 diffuse_brdf(
	vec3 Nn,
	vec3 Ln,
	vec3 Vn,
	vec3 Kd)
{
	return Kd * M_INV_PI;
}

void computeOrtho(vec3 A, out vec3 B, out vec3 C)
{
	B = (abs(A.z) < 0.999) ? vec3(-A.y, A.x, 0.0 )*inversesqrt(1.0-A.z*A.z) :
		vec3(0.0, -A.z, A.y)*inversesqrt(1.0-A.x*A.x) ;
	C = cross( A, B );
}

vec3 irradianceFromSH(vec3 n)
{
	return (shCoefs[0]*n.x + shCoefs[1]*n.y + shCoefs[2]*n.z + shCoefs[3])*n.x
		+ (shCoefs[4]*n.y + shCoefs[5]*n.z + shCoefs[6])*n.y
		+ (shCoefs[7]*n.z + shCoefs[8])*n.z
		+ shCoefs[9];
}

vec3 importanceSampleGGX(vec2 Xi, vec3 A, vec3 B, vec3 C, float roughness)
{
	float a = roughness*roughness;
	float cosT = sqrt((1.0-Xi.y)/(1.0+(a*a-1.0)*Xi.y));
	float sinT = sqrt(1.0-cosT*cosT);
	float phi = 2.0*M_PI*Xi.x;
	return (sinT*cos(phi)) * A + (sinT*sin(phi)) * B + cosT * C;
}

float probabilityGGX(float ndh, float vdh, float Roughness)
{
	if ( UseUE4Shading )
	{
		float a = Roughness * Roughness;
		float a2 = a*a;
		return D_GGX( a2, ndh ) * ndh / (4.0 * vdh);
	}
	else
		return normal_distrib(ndh, Roughness) * ndh / (4.0*vdh);
}

float distortion(vec3 Wn)
{
	// Computes the inverse of the solid angle of the (differential) pixel in
	// the environment map pointed at by Wn
	float sinT = sqrt(1.0-Wn.y*Wn.y);
	return sinT;
}

float computeLOD(vec3 Ln, float p, int nbSamples, float maxLod)
{
	return max(0.0, (maxLod-1.5) - 0.5*(log(float(nbSamples)) + log( p * distortion(Ln) ))
		* M_INV_LOG2);
}

vec3 samplePanoramicLOD(sampler2D map, vec3 dir, float lod)
{
	float n = length(dir.xz);
	vec2 pos = vec2( (n>0.0000001) ? dir.x / n : 0.0, dir.y);
	pos = acos(pos)*M_INV_PI;
	pos.x = (dir.z > 0.0) ? pos.x*0.5 : 1.0-(pos.x*0.5);
	pos.y = 1.0-pos.y;
	return textureLod(map, pos, lod).rgb;
}

vec3 pointLightContribution(
	vec3 fixedNormalWS,
	vec3 pointToLightDirWS,
	vec3 pointToCameraDirWS,
	vec3 diffColor,
	vec3 specColor,
	float roughness,
	vec3 LampColor,
	float LampIntensity,
	float LampDist)
{
	// Note that the lamp intensity is using ˝computer games units" i.e. it needs
	// to be multiplied by M_PI.
	// Cf https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/

	return  max(dot(fixedNormalWS,pointToLightDirWS), 0.0) * ( (
		diffuse_brdf(
			fixedNormalWS,
			pointToLightDirWS,
			pointToCameraDirWS,
			diffColor*(vec3(1.0,1.0,1.0)-specColor))
		+ microfacets_brdf(
			fixedNormalWS,
			pointToLightDirWS,
			pointToCameraDirWS,
			specColor,
			roughness) ) *LampColor*(lampAttenuation(LampDist)*LampIntensity*M_PI) );
}

vec3 dirLightContribution(
	vec3 fixedNormalWS,
	vec3 pointToLightDirWS,
	vec3 pointToCameraDirWS,
	vec3 diffColor,
	vec3 specColor,
	float roughness,
	vec3 LampColor,
	float LampIntensity)
{
	// Note that the lamp intensity is using ˝computer games units" i.e. it needs
	// to be multiplied by M_PI.
	// Cf https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/

	return  max(dot(fixedNormalWS,pointToLightDirWS), 0.0) * ( (
		diffuse_brdf(
			fixedNormalWS,
			pointToLightDirWS,
			pointToCameraDirWS,
			diffColor*(vec3(1.0,1.0,1.0)-specColor))
		+ microfacets_brdf(
			fixedNormalWS,
			pointToLightDirWS,
			pointToCameraDirWS,
			specColor,
			roughness) ) *LampColor*(LampIntensity*M_PI) );
}

void computeSamplingFrame(
	in vec3 iFS_Tangent,
	in vec3 iFS_Binormal,
	in vec3 fixedNormalWS,
	out vec3 Tp,
	out vec3 Bp)
{
	Tp = normalize(iFS_Tangent
		- fixedNormalWS*dot(iFS_Tangent, fixedNormalWS));
	Bp = normalize(iFS_Binormal
		- fixedNormalWS*dot(iFS_Binormal,fixedNormalWS)
		- Tp*dot(iFS_Binormal, Tp));
}

vec2 fibonacci2D(int i, int nbSample)
{
	return vec2(
		float(i+1) * M_GOLDEN_RATIO,
		(float(i)+0.5) / float(nbSamples)
	);
}

vec3 IBLSpecularContribution(
	sampler2D environmentMap,
	float envRotation,
	float maxLod,
	int nbSamples,
	vec3 normalWS,
	vec3 fixedNormalWS,
	vec3 Tp,
	vec3 Bp,
	vec3 pointToCameraDirWS,
	vec3 specColor,
	float roughness)
{
	vec3 sum = vec3(0.0,0.0,0.0);

	float ndv = max( 1e-8, abs(dot( pointToCameraDirWS, fixedNormalWS )) );

	for(int i=0; i<nbSamples; ++i)
	{
		vec2 Xi = fibonacci2D(i, nbSamples);
		vec3 Hn = importanceSampleGGX(Xi,Tp,Bp,fixedNormalWS,roughness);
		vec3 Ln = -reflect(pointToCameraDirWS,Hn);

		float ndl = dot(fixedNormalWS, Ln);

		// Horizon fading trick from http://marmosetco.tumblr.com/post/81245981087
		const float horizonFade = 1.3;
		float horiz = clamp( 1.0 + horizonFade * dot(normalWS, Ln), 0.0, 1.0 );
		horiz *= horiz;
		ndl = max( 1e-8, ndl );

		float vdh = max( 1e-8, abs(dot(pointToCameraDirWS, Hn)) );
		float ndh = max( 1e-8, abs(dot(fixedNormalWS, Hn)) );
		float lodS = roughness < 0.01 ? 0.0 :
			computeLOD(
				Ln,
				probabilityGGX(ndh, vdh, roughness),
				nbSamples,
				maxLod);
		sum +=
			samplePanoramicLOD(environmentMap,rotate(Ln,envRotation),lodS) *
			microfacets_contrib(
				vdh, ndh, ndl, ndv,
				specColor,
				roughness) * horiz;
	}

	return sum / nbSamples;
}

vec3 computeIBL(
	sampler2D environmentMap,
	float envRotation, 
	float maxLod,
	int nbSamples,
	vec3 normalWS,
	vec3 fixedNormalWS,
	vec3 iFS_Tangent,
	vec3 iFS_Binormal,
	vec3 pointToCameraDirWS,
	vec3 diffColor,
	vec3 specColor,
	float roughness,
	float ambientOcclusion)
{
	vec3 Tp,Bp;
	computeSamplingFrame(iFS_Tangent, iFS_Binormal, fixedNormalWS, Tp, Bp);

	vec3 result = IBLSpecularContribution(
		environmentMap,
		envRotation,
		maxLod,
		nbSamples,
		normalWS,
		fixedNormalWS,
		Tp,
		Bp,
		pointToCameraDirWS,
		specColor,
		roughness);

	result += diffColor * (vec3(1.0,1.0,1.0)-specColor) *
		irradianceFromSH(rotate(fixedNormalWS,envRotation));

	return result * ambientOcclusion;
}
