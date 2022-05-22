import Metal

var device: MTLDevice!
var commandQueue: MTLCommandQueue!
var library: MTLLibrary!
var computePipelineState: MTLComputePipelineState!
var threadgroupSize, threadsPerThreadgroup: MTLSize!

var vectorABuffer, vectorBBuffer, outputVectorBuffer: MTLBuffer!
var vectorASharedBuffer, vectorBSharedBuffer, outputVectorSharedBuffer: MTLBuffer!

var vectorLength = 128000000
let computeFunctionName = "vectorAdd"

var vectorA = [Float](repeating: 0, count: vectorLength)
var vectorB = [Float](repeating: 0, count: vectorLength)
var outputVector = [Float](repeating: 0, count: vectorLength)

func setupMetal() {
	device = MTLCreateSystemDefaultDevice()
	commandQueue = device.makeCommandQueue()
	library = device.makeDefaultLibrary()
}

func setupPipeline() {
	if let computeFunction = library.makeFunction(name: computeFunctionName) {
			computePipelineState = try device.makeComputePipelineState(function: computeFunction)

	}
	else {
		fatalError("Kernel functions are not found")
	}
}

func setupThreads() {
	
	let threadgroupWidth = 256
	let groups = (vectorLength + threadgroupWidth - 1) / threadgroupWidth
	
	threadgroupSize = MTLSize(width: groups, height: 1, depth: 1)
	threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)
}

func setupBuffersShared() {
	
	for i in 0 ..< vectorLength {
		vectorB[i] = Float(i) / 100
	}
    for i in 0 ..< vectorLength {
        vectorB[i] = Float(i) / 100
    }
	
	let numberOfBytes = vectorLength * MemoryLayout<Float>.stride
	
	vectorABuffer = device.makeBuffer(bytes: vectorA,
	                                  length: numberOfBytes,
	                                  options: .storageModeShared)
	
	vectorBBuffer = device.makeBuffer(bytes: vectorB,
	                                  length: numberOfBytes,
	                                  options: .storageModeShared)
	
	outputVectorBuffer = device.makeBuffer(bytes: outputVector,
	                                       length: numberOfBytes,
	                                       options: .storageModeShared)
}

func setupBuffersPrivate() {
    
    for i in 0 ..< vectorLength {
        vectorB[i] = Float(i) / 100
    }
    for i in 0 ..< vectorLength {
        vectorB[i] = Float(i) / 100
    }
    
    let numberOfBytes = vectorLength * MemoryLayout<Float>.stride
    
    vectorASharedBuffer =  device.makeBuffer(bytes: vectorA,
                                             length: numberOfBytes,
                                             options: .storageModeShared)
    vectorBSharedBuffer =  device.makeBuffer(bytes: vectorA,
                                             length: numberOfBytes,
                                             options: .storageModeShared)
    outputVectorSharedBuffer = device.makeBuffer(bytes: outputVector,
                                           length: numberOfBytes,
                                           options: .storageModeShared)
    
    vectorABuffer = device.makeBuffer(length: numberOfBytes,
                                      options: .storageModePrivate)
    
    vectorBBuffer = device.makeBuffer(length: numberOfBytes,
                                      options: .storageModePrivate)
    
    outputVectorBuffer = device.makeBuffer(length: numberOfBytes,
                                           options: .storageModePrivate)
}

func setupBuffersManaged() {
    
    for i in 0 ..< vectorLength {
        vectorB[i] = Float(i) / 100
    }
    for i in 0 ..< vectorLength {
        vectorB[i] = Float(i) / 100
    }
    
    let numberOfBytes = vectorLength * MemoryLayout<Float>.stride
    
    vectorABuffer = device.makeBuffer(bytes: vectorA,
                                      length: numberOfBytes,
                                      options: .storageModeManaged)
    
    vectorBBuffer = device.makeBuffer(bytes: vectorB,
                                      length: numberOfBytes,
                                      options: .storageModeManaged)
    
    outputVectorBuffer = device.makeBuffer(bytes: outputVector,
                                           length: numberOfBytes,
                                           options: .storageModeManaged)
}

func copySharedToPrivate() {
    
    let commandBuffer = commandQueue.makeCommandBuffer()
    
    let blitCommandEncoder = commandBuffer?.makeBlitCommandEncoder()
    let numberOfBytes = vectorLength * MemoryLayout<Float>.stride
    let sizeGB = Double(2 * numberOfBytes) / 1000000000.0
    blitCommandEncoder?.copy(from: vectorASharedBuffer, sourceOffset: 0, to: vectorABuffer, destinationOffset: 0, size: numberOfBytes)
    blitCommandEncoder?.copy(from: vectorASharedBuffer, sourceOffset: 0, to: vectorABuffer, destinationOffset: 0, size: numberOfBytes)
    
    blitCommandEncoder?.endEncoding()
    
    commandBuffer?.addCompletedHandler{ cb in
        let bandwidth = sizeGB / (cb.gpuEndTime - cb.gpuStartTime)
        let text = String(format: "shared to private: copy_size: %.2f GB, bandwidth: %.2f GB/ s", sizeGB, bandwidth)
        print(text)
    }
    
    commandBuffer?.commit()
   
    commandBuffer?.waitUntilCompleted()
    
}

func copyPrivateToShared() {
    
    let commandBuffer = commandQueue.makeCommandBuffer()
    
    let blitCommandEncoder = commandBuffer?.makeBlitCommandEncoder()
    let numberOfBytes = vectorLength * MemoryLayout<Float>.stride
    let sizeGB = Double(numberOfBytes) / 1000000000.0
    blitCommandEncoder?.copy(from: outputVectorBuffer, sourceOffset: 0, to: outputVectorSharedBuffer, destinationOffset: 0, size: numberOfBytes)
    
    blitCommandEncoder?.endEncoding()
    
    commandBuffer?.addCompletedHandler{ cb in
        let bandwidth = sizeGB / (cb.gpuEndTime - cb.gpuStartTime)
        let text = String(format: "private to shared: copy_size: %.2f GB, bandwidth: %.2f GB/ s", sizeGB, bandwidth)
        print(text)
    }
    
    commandBuffer?.commit()
    commandBuffer?.waitUntilCompleted()
}

func display(executionDuration: Double) {

    let datasize = vectorLength * MemoryLayout<Float>.stride
    let readsize = 2 * datasize
    let writesize = datasize
    let rwsize_by_GB = (Double(readsize + writesize) / 1000000000)
    let bandwidth = rwsize_by_GB / executionDuration
    let text = String(format: "lendth: %d, rw_size: %.2f GB, bandwidth: %.2f GB/ s, duration: %.3f s", vectorLength, rwsize_by_GB, bandwidth, executionDuration)
    
    print(text)
}

func compute(){
	
	let commandBuffer = commandQueue.makeCommandBuffer()
	
    let computeEncoder = commandBuffer?.makeComputeCommandEncoder()
		
    computeEncoder?.setComputePipelineState(computePipelineState)
	
    computeEncoder?.setBytes(&vectorLength, length: MemoryLayout<Int32>.stride, index: 0)
	
    computeEncoder?.setBuffer(vectorABuffer, offset: 0, index: 1)
    computeEncoder?.setBuffer(vectorBBuffer, offset: 0, index: 2)
    computeEncoder?.setBuffer(outputVectorBuffer, offset: 0, index: 3)
		
    computeEncoder?.dispatchThreadgroups(threadgroupSize,
			threadsPerThreadgroup: threadsPerThreadgroup)
		
    computeEncoder?.endEncoding()
	
  
    commandBuffer?.addCompletedHandler{
        cb in
        display(executionDuration: cb.gpuEndTime - cb.gpuStartTime)
    }
    commandBuffer?.commit()
    let start = Date()
    commandBuffer?.waitUntilCompleted()
    let elapsed = Date().timeIntervalSince(start)
    display(executionDuration: elapsed)
}


setupMetal()

/*
setupBuffersManaged()
setupPipeline()
setupThreads()
compute()
 */

setupBuffersPrivate()
setupPipeline()
setupThreads()
copySharedToPrivate()
compute()
copyPrivateToShared()

