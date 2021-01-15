#define CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

__global__ void ConvolutionKernel(
	const float* devicePaddedImage,
	const unsigned int paddedWidth,
	const unsigned int paddedHeight,
	const float* filter, const int S,
	float* result, const unsigned int width, const unsigned int height)
{
	// Postavlja se velièina filtera na osnovu velièine paddinga
	unsigned int paddingSize = S;
	unsigned int filterSize = 2 * S + 1;
	// Varijabla sum se koristi prilikom normaliziranja raèuna.
	unsigned int sum = 0;

	// Raèunaju se koordinate trenutnog pixela preko infromacije sadržane u bloku tj. threadu koji ga obraðuje
	// Potrebno je i dodati paddingSize jer GPU radi na paddanoj slici
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y + paddingSize;
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x + paddingSize;

	// Osim kod kvadratnih slika (u ovakvom odabiru velièine bloka), nemoguæe je savršeno pokriti sliku blokovima
	// Ova if naredba omoguæava da se raèun ne izvodi u onim threadovima koji su dio bloka koji ne pokriva dio slike
	// Takoðer osigurava da se ignorira okvir slike tj. padding

	if (j >= paddingSize && j < paddedWidth - paddingSize && i >= paddingSize && i < paddedHeight - paddingSize) {
		unsigned int outputPixelPosition = (i - paddingSize) * width + (j - paddingSize);
		result[outputPixelPosition] = 0.0;
		for (int k = -S; k <= S; k++) {
			for (int l = -S; l <= S; l++) {
				unsigned int inputPixelPosition = (i + k) * paddedWidth + (j + l);
				unsigned int coefPos = (k + S) * filterSize + (l + S);
				sum += filter[coefPos];
				result[outputPixelPosition] += devicePaddedImage[inputPixelPosition] * filter[coefPos];
			}
		}
		// Normalizacija
		if (result[outputPixelPosition] > 0.0) {
			result[outputPixelPosition] = result[outputPixelPosition] / sum;
		}
		else
			result[outputPixelPosition] = 0.0;
	}
};

int ZeroPadding(float* fmap, const unsigned int& width, const unsigned int& height,
	const int& filterSize,
	float* paddedImage, const unsigned int& paddedwidth, const unsigned int& paddedheight);

inline unsigned int DivideCieling(const unsigned int& a, const unsigned int& b);

int main()
{
	// Inicijalizacija vektora koji sadrži sliku
	std::vector<unsigned char> image;
	unsigned int width, height;

	// Svaki filter se izvodi iz broja S, S je velièina paddinga
	// Padding je jedan od naèina za popravak rubova konvolucije
	unsigned int S = 1;
	unsigned int filterSize = 2 * S + 1;

	// Uèitavanje slike u formatu RGBARGBARGBA - sirovi podaci
	unsigned error = lodepng::decode(image, width, height, "slika.png");
	// Neuspjelo uèitavanje slike
	if (error) std::cout << "Decoder error -> " << error << ": " << std::endl;

	// Alokacija memorije za spremanje slike u formatu piksela kroz 3 kanala (RGB)
	// Kompletna slika je velièine width * height * 3 kanala
	float* inputImage = new float[(image.size() * 3) / 4];

	// Alokacija memorija za svakoi pojedini kanal
	float* inputImageRED = new float[image.size() / 4];
	float* inputImageGREEN = new float[image.size() / 4];
	float* inputImageBLUE = new float[image.size() / 4];

	float* outputImageR = new float[image.size() / 4];
	float* outputImageG = new float[image.size() / 4];
	float* outputImageB = new float[image.size() / 4];

	// pixelCount govori o kojem se trenutno pikselu radi u RGB nizu. Svako RGBA ponavljanje 
	// je jedan piksel
	int pixelCount = 0;

	// Ignorira se alpha vrijednost jer je bitna samo boja

	for (int i = 0; i < image.size(); i = i + 4) {
		// Pull each RGB pixel to its array
		inputImageRED[pixelCount] = (float)image.at(i);
		inputImageGREEN[pixelCount] = (float)image.at(i + 1);
		inputImageBLUE[pixelCount] = (float)image.at(i + 2);
		pixelCount++;
	}


	// Alokacija memorije za filter na hostu
	float* filter = new float[filterSize * filterSize];
	// Popunjavanje filtera vrijednostima
	filter[0] = 1; filter[1] = 2; filter[2] = 1;
	filter[3] = 2; filter[4] = 4; filter[5] = 2;
	filter[6] = 1; filter[7] = 2; filter[8] = 1;

	// Odabr filtera
	//LoadFilter(filter, filterSize, "GAUSS");

	//Raèunanje paddinga 
	unsigned int paddedWidth = width + 2 * S;
	unsigned int paddedHeight = height + 2 * S;

	// Alokacija memorije za sliku proširenu paddingom
	float* paddedImageR = new float[paddedWidth * paddedHeight];
	float* paddedImageG = new float[paddedWidth * paddedHeight];
	float* paddedImageB = new float[paddedWidth * paddedHeight];

	// Dodavanje paddinga
	ZeroPadding(inputImageRED, width, height, S, paddedImageR, paddedWidth, paddedHeight);
	ZeroPadding(inputImageGREEN, width, height, S, paddedImageG, paddedWidth, paddedHeight);
	ZeroPadding(inputImageBLUE, width, height, S, paddedImageB, paddedWidth, paddedHeight);


	// Alokacija memorije za sliku na device-u i transfer paddane slike sa hosta na device
	float* devicePaddedImageR;
	float* devicePaddedImageG;
	float* devicePaddedImageB;

	unsigned int paddedImageSizeByte = paddedWidth * paddedHeight * sizeof(float);

	cudaMalloc(reinterpret_cast<void**>(&devicePaddedImageR), paddedImageSizeByte);
	cudaMemcpy(devicePaddedImageR, paddedImageR, paddedImageSizeByte, cudaMemcpyHostToDevice);
	cudaMalloc(reinterpret_cast<void**>(&devicePaddedImageG), paddedImageSizeByte);
	cudaMemcpy(devicePaddedImageG, paddedImageG, paddedImageSizeByte, cudaMemcpyHostToDevice);
	cudaMalloc(reinterpret_cast<void**>(&devicePaddedImageB), paddedImageSizeByte);
	cudaMemcpy(devicePaddedImageB, paddedImageB, paddedImageSizeByte, cudaMemcpyHostToDevice);// Host to Device

	// Alokacija i transfer filtera na device
	float* deviceFilter;
	unsigned int filterKernelSizeByte = filterSize * filterSize * sizeof(float);
	cudaMalloc(reinterpret_cast<void**>(&deviceFilter), filterKernelSizeByte);
	cudaMemcpy(deviceFilter, filter, filterKernelSizeByte, cudaMemcpyHostToDevice);


	// Postavljanje konfiguracije izvedbe
	// Koriste se blokovi velièine 16x16 da se osigura dovoljan broj threadova
	// koji æe izvršavati zadatke. U ovom programu se ne zna kojoj grafièkoj kartici se izvodi 
	// program pa je 16x16 zlatna sredina. Bitno je da blok bude velièine
	// potencije broja 2
	const unsigned int blockWidth = 16;
	const unsigned int blockHeight = 16;

	// Funkcija DivideCeiling vraæa veæu vrijednost prilikom cjelobrojnog dijeljenja
	// Potrebno je izraèunati grid tj. podijeliti ulaznu sliku na blokove, a te blokove
	// na threadove koji æe raèunati konvoluciju na svakom pikselu slike
	// Velièina grida je width/blockWidth X height/blockHeight = 120 x 68 blokova
	// Velièina bloka je 16x16 threadova
	// Sve skupa 2088960 threadova izvršavanja. Svaki piksel dobiva svoj thread izvršavanja
	// Naravno, što je bolja grafièka kartica bit æe moguæe izvršiti više threadova
	// istovremeno.

	const dim3 grid(DivideCieling(width, blockWidth), DivideCieling(height, blockHeight));
	const dim3 threadBlock(blockWidth, blockHeight);

	// Alokacija i transfer memorije na GPU koja sadržava rezultat konvolucije
	// Bitno je konvoluciju raditi na kompletno novoj slici radi kazualnosti tj.
	// da nam veæ izraèunati pikseli ne utjeèu na neizraèunate
	float* deviceResultR;
	float* deviceResultG;
	float* deviceResultB;
	unsigned int imageSizeByte = width * height * sizeof(float);
	
	cudaMalloc(reinterpret_cast<void**>(&deviceResultR), imageSizeByte);
	cudaMalloc(reinterpret_cast<void**>(&deviceResultG), imageSizeByte);
	cudaMalloc(reinterpret_cast<void**>(&deviceResultB), imageSizeByte);

	// Poziv kernel funkcije
	ConvolutionKernel <<<grid, threadBlock >>> (devicePaddedImageR, paddedWidth, paddedHeight, deviceFilter, S, deviceResultR, width, height);
	ConvolutionKernel <<<grid, threadBlock >>> (devicePaddedImageG, paddedWidth, paddedHeight, deviceFilter, S, deviceResultG, width, height);
	ConvolutionKernel <<<grid, threadBlock >>> (devicePaddedImageB, paddedWidth, paddedHeight, deviceFilter, S, deviceResultB, width, height);
	// Kopiranje memorije natrag na host
	cudaMemcpy(outputImageR, deviceResultR, imageSizeByte, cudaMemcpyDeviceToHost); 
	cudaMemcpy(outputImageG, deviceResultG, imageSizeByte, cudaMemcpyDeviceToHost); 
	cudaMemcpy(outputImageB, deviceResultB, imageSizeByte, cudaMemcpyDeviceToHost); 

	// Bitno je èekati da se svi izraèuni izvrše i za to se koristi sinkronizacijska funkcija
	// U suprotnom, host tj. CPU bi zapoèeo dekodiranje slike za koju još nisu gotovi izraèuni jer
	// CPU i GPU mogu raditi odvojeno zadatke
	cudaDeviceSynchronize();


	// Spajanje RGB kanala u jednu sliku. Dodaje se i alpha kanal zbog .png formata
	std::vector<unsigned char> outputImage;

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			outputImage.push_back(outputImageR[i * height + j]);
			outputImage.push_back(outputImageG[i * height + j]);
			outputImage.push_back(outputImageB[i * height + j]);
			outputImage.push_back(255); 
		}
	}

	// Enkodiranje i spremanje slike na disk
	error = lodepng::encode("konvolucija.png", outputImage, width, height);
	if (error) std::cout << "encoder error " << error << ": " << std::endl;

	delete[] inputImage;
	return 0;
}


int ZeroPadding(float* inputImage, const unsigned int& width, const unsigned int& height, const int& filterSize,
	float* paddedImage, const unsigned int& paddedwidth, const unsigned int& paddedheight)
{

	if (paddedImage == NULL) printf("wtf?");

	for (unsigned int i = 0; i < paddedheight; i++) {
		for (unsigned int j = 0; j < paddedwidth; j++) {

			// Set the pixel position of the padded fmap
			unsigned int paddedPixelPos = i * paddedwidth + j;

			// Copy the pixel value
			if (i >= filterSize && i < height + filterSize &&
				j >= filterSize && j < width + filterSize) {
				unsigned int pixelPos = (i - filterSize) * width + (j - filterSize);
				paddedImage[paddedPixelPos] = inputImage[pixelPos];
			}
			else {
				paddedImage[paddedPixelPos] = 0;
			}
		}
	}

	return 0;
};

inline unsigned int DivideCieling(const unsigned int& a, const unsigned int& b) { return (a % b != 0) ? (a / b + 1) : (a / b); };

//int LoadFilter(float* filter, int filterSize, std::string filterName) {
//	if (filterName == "GAUSS") {
//		for (int i = 0; i < filterSize * filterSize; i++) {
//			for (int j = 0; j < filterSize * filterSize; j++)
//				*(filter + 2 * i + j) = gaussBlur[i][j];
//			return 0;
//		}
//	}
//}

