import SwiftUI
import FieldAtlas

@main
struct FieldAtlasApp: App {
    var body: some Scene {
        WindowGroup {
            AtlasView()
                .ignoresSafeArea()
        }
    }
}

struct AtlasView: UIViewControllerRepresentable {
    func makeUIViewController(
        context: Context
    ) -> UIViewController {
        NativeCaptureKt.MainViewController()
    }

    func updateUIViewController(
        _ controller: UIViewController,
        context: Context
    ) {}
}
