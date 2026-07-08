// WelcomeViewModel drives the first-run wizard. The interesting behaviors
// are validation gates (storage + api-key) feeding `lastError`, the
// intro → setup → hardwareDetect → recommend → complete state, and
// the Start Server validation path.

import XCTest
@testable import FusionMLX

@MainActor
final class WelcomeViewModelTests: XCTestCase {

    // AppServices uses a weak reference to its services on WelcomeViewModel,
    // so the test must keep a strong reference for the lifetime of each case.
    private var services: AppServices!

    private func makeVM(basePath: String = "/Users/Fido/.fusion",
                        modelDir: String  = "/Users/Fido/.fusion/models",
                        port: Int = 8000,
                        apiKey: String? = nil) -> WelcomeViewModel {
        let cfg = AppConfig(
            bindAddress: "127.0.0.1",
            port: port,
            apiKey: apiKey,
            basePath: basePath,
            modelDir: modelDir,
            hfEndpoint: ""
        )
        services = AppServices(config: cfg, server: nil)
        return WelcomeViewModel(services: services, server: nil)
    }

    // MARK: - flow

    func testStartsOnIntroStep() {
        let vm = makeVM()
        XCTAssertEqual(vm.step, .intro)
    }

    func testBeginSetupAdvancesToSetupAndClearsError() {
        let vm = makeVM()
        vm.apiKey = "abc"
        XCTAssertFalse(vm.validateApiKey())
        XCTAssertNotNil(vm.lastError)
        vm.beginSetup()
        XCTAssertEqual(vm.step, .setup)
        XCTAssertNil(vm.lastError)
    }

    func testDefaultPortIs8000() {
        let vm = makeVM()
        XCTAssertEqual(vm.portText, "8000")
    }

    // MARK: - validateSetup

    func testValidateSetupHappyPath() {
        let vm = makeVM()
        vm.apiKey = "secret-key"
        XCTAssertTrue(vm.validateSetup())
        XCTAssertNil(vm.lastError)
    }

    func testValidateSetupFailsOnEmptyBase() {
        let vm = makeVM()
        vm.basePath = "   "
        vm.apiKey = "secret-key"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "Base directory is required.")
    }

    func testValidateSetupFailsOnInvalidPort() {
        let vm = makeVM()
        vm.apiKey = "secret-key"
        vm.portText = "0"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "Port must be a number between 1 and 65535.")
    }

    func testValidateSetupFailsOnPortNonNumeric() {
        let vm = makeVM()
        vm.apiKey = "secret-key"
        vm.portText = "abc"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "Port must be a number between 1 and 65535.")
    }

    func testValidateSetupFailsOnShortApiKey() {
        let vm = makeVM()
        vm.apiKey = "abc"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "API key must be at least 4 characters.")
    }

    func testValidateSetupFailsOnApiKeyWhitespace() {
        let vm = makeVM()
        // 4+ chars but a space inside — server-side validator rejects.
        vm.apiKey = "ab cd"
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "API key must not contain whitespace.")
    }

    func testValidateSetupFailsOnApiKeyNonPrintable() {
        let vm = makeVM()
        vm.apiKey = "abcd\u{007F}"   // DEL char, outside printable ASCII
        XCTAssertFalse(vm.validateSetup())
        XCTAssertEqual(vm.lastError, "API key must contain only printable ASCII.")
    }

    // MARK: - whichllm / hardware detection flow

    func testBeginHardwareDetectionAdvancesToHardwareStep() {
        let vm = makeVM()
        vm.beginHardwareDetection()
        XCTAssertEqual(vm.step, .hardwareDetect)
    }

    func testBeginRecommendationAdvancesToRecommendStep() {
        let vm = makeVM()
        vm.beginRecommendation()
        XCTAssertEqual(vm.step, .recommend)
    }

    func testHardwareInfoDefaultsAvailable() {
        let vm = makeVM()
        // quickDetect provides basic hardware info from ProcessInfo
        let hw = vm.hardwareInfo
        XCTAssertGreaterThan(hw.cpuCores, 0)
        XCTAssertGreaterThan(hw.ramGB, 0)
    }

    func testRecommendedSettingsAdaptToHardware() {
        let vm = makeVM()

        // Simulate Apple Silicon with large RAM — use exact GiB values
        let gb128 = 128 * 1024 * 1024 * 1024  // 128 GiB bytes
        let gb500 = 500 * 1000 * 1000 * 1000   // 500 GB disk
        let appleHw = HardwareInfoDTO(
            gpus: [GPUInfoDTO(
                name: "Apple M5 Max", vendor: "apple",
                vramBytes: gb128, usableVramBytes: nil,
                memoryBandwidthGbps: 614.0, sharedMemory: true
            )],
            cpu: "Apple M5 Max", cpuCores: 18,
            ramBytes: gb128, ramBudgetBytes: nil,
            budgetNotes: nil, diskFreeBytes: gb500,
            os: "darwin"
        )
        vm.hardwareInfo = appleHw

        XCTAssertEqual(vm.hardwareInfo.ramGB, 128.0, accuracy: 0.01)
    }

    func testNavigationBackFromHardwareDetect() {
        let vm = makeVM()
        vm.beginHardwareDetection()
        XCTAssertEqual(vm.step, .hardwareDetect)
        vm.backToSetup()
        XCTAssertEqual(vm.step, .setup)
    }

    func testNavigationBackFromRecommend() {
        let vm = makeVM()
        vm.beginRecommendation()
        XCTAssertEqual(vm.step, .recommend)
        vm.backToHardwareDetect()
        XCTAssertEqual(vm.step, .hardwareDetect)
    }

    func testModelRecommendationDefaultState() {
        let vm = makeVM()
        XCTAssertTrue(vm.recommendations.isEmpty)
        XCTAssertEqual(vm.selectedModelIndex, 0)
        XCTAssertNil(vm.recommendError)
        XCTAssertFalse(vm.isRecommending)
    }

    func testWhichLLMDTOEncoding() throws {
        let model = ModelRecommendationDTO(
            rank: 1,
            modelId: "test/TestModel",
            modelName: "Test Model",
            artifactRepoId: nil,
            artifactFilename: nil,
            parameterCount: 7_000_000_000,
            parameterCountActive: nil,
            architecture: "llama",
            contextLength: 8192,
            quantType: "Q4_K_M",
            fileSizeBytes: 4_000_000_000,
            vramRequiredBytes: 6_000_000_000,
            vramAvailableBytes: 128_000_000_000,
            estimatedTokPerSec: 45.5,
            speedConfidence: "medium",
            qualityScore: 85.3,
            fitType: "full_gpu",
            benchmarkStatus: "direct",
            usesMultiGpu: false
        )
        XCTAssertEqual(model.paramsB, 7.0, accuracy: 0.01)
        XCTAssertEqual(model.fileSizeGB, 3.73, accuracy: 0.01)  // binary GiB: 4e9 / 1024^3
        XCTAssertEqual(model.fitDescription, "Full GPU")
        XCTAssertEqual(model.id, "test/TestModel")
    }
}
