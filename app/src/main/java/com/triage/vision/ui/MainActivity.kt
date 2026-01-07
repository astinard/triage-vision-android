package com.triage.vision.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.triage.vision.camera.DepthCameraManager
import com.triage.vision.camera.DepthFrameSynchronizer
import com.triage.vision.data.ObservationEntity
import com.triage.vision.pipeline.FastPipeline
import com.triage.vision.pipeline.SlowPipeline
import com.triage.vision.ui.theme.TriageVisionTheme
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA
        )
    }

    private val viewModel: MonitoringViewModel by viewModels()
    private lateinit var cameraExecutor: ExecutorService

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.all { it.value }
        if (allGranted) {
            Log.i(TAG, "All permissions granted")
        } else {
            Log.w(TAG, "Some permissions denied")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (!hasPermissions()) {
            permissionLauncher.launch(REQUIRED_PERMISSIONS)
        }

        setContent {
            TriageVisionTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen(
                        viewModel = viewModel,
                        cameraExecutor = cameraExecutor
                    )
                }
            }
        }
    }

    private fun hasPermissions(): Boolean {
        return REQUIRED_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    viewModel: MonitoringViewModel,
    cameraExecutor: ExecutorService
) {
    val uiState by viewModel.uiState.collectAsStateWithLifecycle()
    val observations by viewModel.recentObservations.collectAsStateWithLifecycle(initialValue = emptyList())

    var selectedTab by remember { mutableIntStateOf(0) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Triage Vision") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    titleContentColor = MaterialTheme.colorScheme.onPrimary
                ),
                actions = {
                    StatusIndicator(
                        isMonitoring = uiState.isMonitoring,
                        isAnalyzing = uiState.isAnalyzing
                    )
                }
            )
        },
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    icon = { Text("📹") },
                    label = { Text("Monitor") },
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 }
                )
                NavigationBarItem(
                    icon = { Text("📋") },
                    label = { Text("Charts") },
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 }
                )
                NavigationBarItem(
                    icon = { Text("⚙️") },
                    label = { Text("Settings") },
                    selected = selectedTab == 2,
                    onClick = { selectedTab = 2 }
                )
            }
        }
    ) { padding ->
        Box(modifier = Modifier.padding(padding)) {
            when (selectedTab) {
                0 -> MonitoringScreen(
                    uiState = uiState,
                    viewModel = viewModel,
                    cameraExecutor = cameraExecutor
                )
                1 -> ChartHistoryScreen(observations = observations)
                2 -> SettingsScreen()
            }
        }

        uiState.currentAlert?.let { alert ->
            AlertDialog(
                onDismissRequest = { viewModel.acknowledgeAlert() },
                title = { Text("Alert") },
                text = {
                    Text(
                        when (alert) {
                            is FastPipeline.Alert.FallDetected ->
                                "FALL DETECTED - Immediate attention required"
                            is FastPipeline.Alert.DepthVerifiedFall ->
                                "FALL DETECTED (Verified) - Drop: %.1fm, Confidence: %.0f%%".format(
                                    alert.dropMeters,
                                    alert.confidence * 100
                                )
                            is FastPipeline.Alert.LeavingBedZone ->
                                "Patient leaving bed zone - Distance: %.1fm".format(
                                    alert.distanceMeters
                                )
                            is FastPipeline.Alert.Stillness ->
                                "Patient has been still for ${alert.durationSeconds / 60} minutes"
                            is FastPipeline.Alert.PoseChange ->
                                "Position changed: ${alert.from} → ${alert.to}"
                        }
                    )
                },
                confirmButton = {
                    Button(onClick = { viewModel.acknowledgeAlert() }) {
                        Text("Acknowledge")
                    }
                }
            )
        }

        uiState.error?.let { error ->
            Snackbar(
                action = {
                    TextButton(onClick = { viewModel.clearError() }) {
                        Text("Dismiss")
                    }
                }
            ) {
                Text(error)
            }
        }
    }
}

@Composable
fun StatusIndicator(isMonitoring: Boolean, isAnalyzing: Boolean) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(end = 16.dp)
    ) {
        if (isAnalyzing) {
            CircularProgressIndicator(
                modifier = Modifier.size(20.dp),
                color = MaterialTheme.colorScheme.onPrimary,
                strokeWidth = 2.dp
            )
        }
        Text(
            text = if (isMonitoring) "● LIVE" else "○ OFF",
            color = if (isMonitoring)
                MaterialTheme.colorScheme.onPrimary
            else
                MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.5f)
        )
    }
}

@Composable
fun MonitoringScreen(
    uiState: MonitoringViewModel.UiState,
    viewModel: MonitoringViewModel,
    cameraExecutor: ExecutorService
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // Initialize depth camera manager
    val depthCameraManager = remember { DepthCameraManager(context) }
    val frameSynchronizer = remember { DepthFrameSynchronizer() }

    var depthAvailable by remember { mutableStateOf(false) }

    // Check for depth camera and start it
    LaunchedEffect(Unit) {
        // Initialize first, then check support
        depthCameraManager.initialize()
        depthAvailable = depthCameraManager.isDepthSupported()
        viewModel.setDepthEnabled(depthAvailable)

        if (depthAvailable) {
            Log.i("MonitoringScreen", "Depth camera available, starting...")
            depthCameraManager.start()

            // Collect depth frames and feed to synchronizer
            scope.launch {
                depthCameraManager.depthFrames.collect { depthFrame ->
                    frameSynchronizer.onDepthFrame(depthFrame)
                }
            }

            // Collect synchronized frames
            scope.launch {
                frameSynchronizer.syncedFrames.collect { syncedFrame ->
                    if (uiState.isMonitoring) {
                        viewModel.processFrameWithDepth(syncedFrame.rgb, syncedFrame.depth)
                    }
                }
            }
        } else {
            Log.i("MonitoringScreen", "No depth camera available, using RGB-only")
        }
    }

    // Cleanup on dispose
    DisposableEffect(Unit) {
        onDispose {
            depthCameraManager.stop()
            frameSynchronizer.clear()
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
        ) {
            CameraPreview(
                modifier = Modifier.fillMaxSize(),
                cameraExecutor = cameraExecutor,
                onFrameAnalyzed = { bitmap, timestamp ->
                    if (depthAvailable) {
                        // Feed RGB frame to synchronizer
                        frameSynchronizer.onRgbFrame(bitmap, timestamp)
                    } else if (uiState.isMonitoring) {
                        // Fallback: RGB-only processing
                        viewModel.processFrame(bitmap)
                    }
                }
            )

            DetectionOverlay(
                personDetected = uiState.personDetected,
                pose = uiState.currentPose,
                motionLevel = uiState.motionLevel,
                fps = uiState.fps,
                depthEnabled = uiState.depthEnabled,
                distanceMeters = uiState.distanceMeters,
                inBedZone = uiState.inBedZone
            )
        }

        ControlPanel(
            isMonitoring = uiState.isMonitoring,
            isAnalyzing = uiState.isAnalyzing,
            secondsSinceMotion = uiState.secondsSinceMotion,
            lastObservation = uiState.lastObservation,
            onStartMonitoring = { viewModel.startMonitoring() },
            onStopMonitoring = { viewModel.stopMonitoring() },
            onAnalyzeNow = { viewModel.triggerVlmAnalysis() }
        )
    }
}

@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    cameraExecutor: ExecutorService,
    onFrameAnalyzed: (Bitmap, Long) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember { PreviewView(context) }

    LaunchedEffect(Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetResolution(Size(640, 480))
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        val bitmap = imageProxy.toBitmap()
                        val timestamp = imageProxy.imageInfo.timestamp
                        onFrameAnalyzed(bitmap, timestamp)
                        imageProxy.close()
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
            } catch (e: Exception) {
                Log.e("CameraPreview", "Camera bind failed", e)
            }

        }, ContextCompat.getMainExecutor(context))
    }

    AndroidView(
        factory = { previewView },
        modifier = modifier
    )
}

@Composable
fun DetectionOverlay(
    personDetected: Boolean,
    pose: FastPipeline.Pose,
    motionLevel: Float,
    fps: Float,
    depthEnabled: Boolean = false,
    distanceMeters: Float = 0f,
    inBedZone: Boolean = false
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Row(
            horizontalArrangement = Arrangement.SpaceBetween,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = "%.1f FPS".format(fps),
                color = Color.White,
                style = MaterialTheme.typography.labelSmall
            )
            if (depthEnabled) {
                Text(
                    text = "DEPTH",
                    color = Color.Cyan,
                    style = MaterialTheme.typography.labelSmall
                )
            }
        }

        Spacer(modifier = Modifier.weight(1f))

        Surface(
            color = MaterialTheme.colorScheme.surface.copy(alpha = 0.8f),
            shape = MaterialTheme.shapes.medium
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(
                    text = if (personDetected) "Person Detected" else "No Person",
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (personDetected)
                        MaterialTheme.colorScheme.primary
                    else
                        MaterialTheme.colorScheme.outline
                )
                Text(
                    text = "Pose: ${pose.name}",
                    style = MaterialTheme.typography.bodySmall
                )
                Text(
                    text = "Motion: %.0f%%".format(motionLevel * 100),
                    style = MaterialTheme.typography.bodySmall
                )

                // Depth metrics
                if (depthEnabled && personDetected) {
                    HorizontalDivider(
                        modifier = Modifier.padding(vertical = 4.dp),
                        color = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
                    )
                    Text(
                        text = "Distance: %.1fm".format(distanceMeters),
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.Cyan
                    )
                    Text(
                        text = if (inBedZone) "In Bed Zone" else "Out of Bed Zone",
                        style = MaterialTheme.typography.bodySmall,
                        color = if (inBedZone)
                            MaterialTheme.colorScheme.primary
                        else
                            MaterialTheme.colorScheme.error
                    )
                }
            }
        }
    }
}

@Composable
fun ControlPanel(
    isMonitoring: Boolean,
    isAnalyzing: Boolean,
    secondsSinceMotion: Long,
    lastObservation: SlowPipeline.Observation?,
    onStartMonitoring: () -> Unit,
    onStopMonitoring: () -> Unit,
    onAnalyzeNow: () -> Unit
) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        tonalElevation = 4.dp
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = if (secondsSinceMotion > 60)
                        "Still for ${secondsSinceMotion / 60}m"
                    else
                        "Active",
                    style = MaterialTheme.typography.bodyMedium
                )

                lastObservation?.let {
                    Text(
                        text = "Last: ${it.position}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.outline
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Button(
                    onClick = { if (isMonitoring) onStopMonitoring() else onStartMonitoring() },
                    modifier = Modifier.weight(1f),
                    colors = if (isMonitoring)
                        ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
                    else
                        ButtonDefaults.buttonColors()
                ) {
                    Text(if (isMonitoring) "Stop" else "Start")
                }

                OutlinedButton(
                    onClick = onAnalyzeNow,
                    modifier = Modifier.weight(1f),
                    enabled = isMonitoring && !isAnalyzing
                ) {
                    if (isAnalyzing) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            strokeWidth = 2.dp
                        )
                    } else {
                        Text("Analyze")
                    }
                }
            }
        }
    }
}

@Composable
fun ChartHistoryScreen(observations: List<ObservationEntity>) {
    if (observations.isEmpty()) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "No observations recorded yet",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.outline
            )
        }
    } else {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(observations) { observation ->
                ObservationCard(observation = observation)
            }
        }
    }
}

@Composable
fun ObservationCard(observation: ObservationEntity) {
    val timestamp = remember(observation.timestamp) {
        SimpleDateFormat("MMM d, HH:mm", Locale.getDefault())
            .format(Date(observation.timestamp))
    }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = timestamp,
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.outline
                )
                Text(
                    text = observation.triggeredBy,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.secondary
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Position: ${observation.position.replace("_", " ")}",
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                text = "Alertness: ${observation.alertness.replace("_", " ")}",
                style = MaterialTheme.typography.bodySmall
            )

            if (observation.chartNote.isNotBlank()) {
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = observation.chartNote,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
fun SettingsScreen() {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text(
            text = "Settings",
            style = MaterialTheme.typography.headlineMedium
        )

        Spacer(modifier = Modifier.height(24.dp))

        SettingsItem(title = "VLM Analysis Interval", value = "15 minutes")
        SettingsItem(title = "Stillness Alert Threshold", value = "30 minutes")
        SettingsItem(title = "Fall Detection", value = "Enabled")
        SettingsItem(title = "Data Retention", value = "24 hours")
    }
}

@Composable
fun SettingsItem(title: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 12.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(text = title)
        Text(text = value, color = MaterialTheme.colorScheme.outline)
    }
    HorizontalDivider()
}
