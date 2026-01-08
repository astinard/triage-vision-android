package com.triage.vision.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Build
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
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.triage.vision.camera.DepthCameraManager
import com.triage.vision.camera.DepthFrameSynchronizer
import com.triage.vision.classifier.NursingLabels
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
        private val REQUIRED_PERMISSIONS = buildList {
            add(Manifest.permission.CAMERA)
            // Android 13+ requires POST_NOTIFICATIONS
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                add(Manifest.permission.POST_NOTIFICATIONS)
            }
        }.toTypedArray()
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
            Log.w(TAG, "Some permissions denied: ${permissions.filter { !it.value }.keys}")
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

    override fun onStart() {
        super.onStart()
        // Bind to service if it's running
        viewModel.bindToService(this)
    }

    override fun onStop() {
        super.onStop()
        // Unbind from service (but don't stop it)
        viewModel.unbindFromService(this)
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
    val context = LocalContext.current

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
                        isAnalyzing = uiState.isAnalyzing,
                        isServiceBound = uiState.isServiceBound
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
                    cameraExecutor = cameraExecutor,
                    onStartMonitoring = { viewModel.startMonitoring(context) },
                    onStopMonitoring = { viewModel.stopMonitoring(context) }
                )
                1 -> ChartHistoryScreen(observations = observations)
                2 -> SettingsScreen(
                    useBackgroundService = uiState.useBackgroundService,
                    onBackgroundServiceToggle = { viewModel.setBackgroundServiceEnabled(it) }
                )
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
fun StatusIndicator(
    isMonitoring: Boolean,
    isAnalyzing: Boolean,
    isServiceBound: Boolean = false
) {
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

        // Service indicator
        if (isServiceBound) {
            Text(
                text = "SVC",
                color = Color.Cyan,
                style = MaterialTheme.typography.labelSmall
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
    cameraExecutor: ExecutorService,
    onStartMonitoring: () -> Unit,
    onStopMonitoring: () -> Unit
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
                    if (uiState.isMonitoring && !uiState.useBackgroundService) {
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
            // Always show camera preview when monitoring (even with background service)
            CameraPreview(
                modifier = Modifier.fillMaxSize(),
                cameraExecutor = cameraExecutor,
                onFrameAnalyzed = { bitmap, timestamp ->
                    // Always send frames to ViewModel - it will decide whether to process
                    // This handles the case where isMonitoring state hasn't propagated yet
                    if (depthAvailable && !uiState.useBackgroundService && uiState.isMonitoring) {
                        // Feed RGB frame to synchronizer (foreground only mode with depth)
                        frameSynchronizer.onRgbFrame(bitmap, timestamp)
                    } else {
                        // Send to ViewModel - it routes to service or processes locally
                        viewModel.processFrame(bitmap)
                    }
                }
            )

            // Skeleton overlay
            if (uiState.landmarks.isNotEmpty()) {
                SkeletonOverlay(
                    landmarks = uiState.landmarks,
                    modifier = Modifier.fillMaxSize()
                )
            }

            DetectionOverlay(
                personDetected = uiState.personDetected,
                pose = uiState.currentPose,
                poseConfidence = uiState.poseConfidence,
                motionLevel = uiState.motionLevel,
                fps = uiState.fps,
                depthEnabled = uiState.depthEnabled,
                distanceMeters = uiState.distanceMeters,
                inBedZone = uiState.inBedZone,
                isBackgroundMode = uiState.useBackgroundService && uiState.isMonitoring,
                lastObservation = uiState.lastObservation,
                isAnalyzing = uiState.isAnalyzing,
                // CLIP classification
                clipEnabled = uiState.clipEnabled,
                clipPosition = uiState.clipPosition,
                clipPositionConfidence = uiState.clipPositionConfidence,
                clipAlertness = uiState.clipAlertness,
                clipAlertnessConfidence = uiState.clipAlertnessConfidence,
                clipActivity = uiState.clipActivity,
                clipActivityConfidence = uiState.clipActivityConfidence,
                clipSafety = uiState.clipSafety,
                clipSafetyConfidence = uiState.clipSafetyConfidence,
                clipInferenceMs = uiState.clipInferenceMs
            )
        }

        ControlPanel(
            isMonitoring = uiState.isMonitoring,
            isAnalyzing = uiState.isAnalyzing,
            secondsSinceMotion = uiState.secondsSinceMotion,
            lastObservation = uiState.lastObservation,
            useBackgroundService = uiState.useBackgroundService,
            isServiceBound = uiState.isServiceBound,
            onStartMonitoring = onStartMonitoring,
            onStopMonitoring = onStopMonitoring,
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

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

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

/**
 * Skeleton overlay - draws stick figure on pose landmarks
 */
@Composable
fun SkeletonOverlay(
    landmarks: List<FastPipeline.PoseLandmark>,
    modifier: Modifier = Modifier
) {
    // MediaPipe pose connections (pairs of landmark indices)
    val connections = listOf(
        // Torso
        11 to 12, 11 to 23, 12 to 24, 23 to 24,
        // Left arm
        11 to 13, 13 to 15,
        // Right arm
        12 to 14, 14 to 16,
        // Left leg
        23 to 25, 25 to 27,
        // Right leg
        24 to 26, 26 to 28,
        // Face
        0 to 1, 1 to 2, 2 to 3, 3 to 7,
        0 to 4, 4 to 5, 5 to 6, 6 to 8
    )

    Canvas(modifier = modifier) {
        val width = size.width
        val height = size.height

        // Draw connections (skeleton lines)
        connections.forEach { (startIdx, endIdx) ->
            if (startIdx < landmarks.size && endIdx < landmarks.size) {
                val start = landmarks[startIdx]
                val end = landmarks[endIdx]

                // Only draw if both landmarks are visible
                if (start.visibility > 0.5f && end.visibility > 0.5f) {
                    drawLine(
                        color = Color.Green,
                        start = Offset(start.x * width, start.y * height),
                        end = Offset(end.x * width, end.y * height),
                        strokeWidth = 4f,
                        cap = StrokeCap.Round
                    )
                }
            }
        }

        // Draw landmark points
        landmarks.forEachIndexed { index, landmark ->
            if (landmark.visibility > 0.5f) {
                val x = landmark.x * width
                val y = landmark.y * height

                // Different colors for different body parts
                val color = when {
                    index in 0..10 -> Color.Yellow  // Face
                    index in 11..22 -> Color.Cyan   // Arms
                    else -> Color.Magenta           // Legs
                }

                drawCircle(
                    color = color,
                    radius = 6f,
                    center = Offset(x, y)
                )
            }
        }
    }
}

@Composable
fun DetectionOverlay(
    personDetected: Boolean,
    pose: FastPipeline.Pose,
    poseConfidence: Float = 0f,
    motionLevel: Float,
    fps: Float,
    depthEnabled: Boolean = false,
    distanceMeters: Float = 0f,
    inBedZone: Boolean = false,
    isBackgroundMode: Boolean = false,
    lastObservation: SlowPipeline.Observation? = null,
    isAnalyzing: Boolean = false,
    // CLIP classification
    clipEnabled: Boolean = false,
    clipPosition: NursingLabels.Position = NursingLabels.Position.LYING_SUPINE,
    clipPositionConfidence: Float = 0f,
    clipAlertness: NursingLabels.Alertness = NursingLabels.Alertness.EYES_CLOSED,
    clipAlertnessConfidence: Float = 0f,
    clipActivity: NursingLabels.Activity = NursingLabels.Activity.STILL,
    clipActivityConfidence: Float = 0f,
    clipSafety: NursingLabels.SafetyConcern = NursingLabels.SafetyConcern.NONE,
    clipSafetyConfidence: Float = 0f,
    clipInferenceMs: Long = 0
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // Top status bar
        Row(
            horizontalArrangement = Arrangement.SpaceBetween,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = "%.1f FPS".format(fps),
                color = Color.White,
                style = MaterialTheme.typography.labelSmall
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (isAnalyzing) {
                    Text(
                        text = "VLM...",
                        color = Color.Magenta,
                        style = MaterialTheme.typography.labelSmall
                    )
                }
                if (isBackgroundMode) {
                    Text(
                        text = "BG",
                        color = Color.Green,
                        style = MaterialTheme.typography.labelSmall
                    )
                }
                if (depthEnabled) {
                    Text(
                        text = "DEPTH",
                        color = Color.Cyan,
                        style = MaterialTheme.typography.labelSmall
                    )
                }
            }
        }

        Spacer(modifier = Modifier.weight(1f))

        // Detection metrics
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
                    text = if (poseConfidence > 0)
                        "Pose: ${pose.name} (${(poseConfidence * 100).toInt()}%)"
                    else
                        "Pose: ${pose.name}",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (pose != FastPipeline.Pose.UNKNOWN) Color.Green else Color.Gray
                )
                Text(
                    text = "Motion: %.0f%%".format(motionLevel * 100),
                    style = MaterialTheme.typography.bodySmall
                )
                Text(
                    text = if (motionLevel < 0.1f) "STILL" else "MOVING",
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Bold,
                    color = if (motionLevel < 0.1f) Color.Yellow else Color.Green
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

        // CLIP Real-time Classification Panel
        if (clipEnabled && personDetected) {
            Spacer(modifier = Modifier.height(8.dp))
            Surface(
                color = Color(0xFF1A1A2E).copy(alpha = 0.9f),
                shape = MaterialTheme.shapes.medium
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(
                        horizontalArrangement = Arrangement.SpaceBetween,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = "CLIP Classification",
                            style = MaterialTheme.typography.labelMedium,
                            color = Color(0xFF00D9FF),
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "${clipInferenceMs}ms",
                            style = MaterialTheme.typography.labelSmall,
                            color = Color.Gray
                        )
                    }
                    Spacer(modifier = Modifier.height(6.dp))

                    // Position
                    ClipClassificationRow(
                        label = "Position",
                        value = clipPosition.label.replace("_", " "),
                        confidence = clipPositionConfidence,
                        color = Color(0xFF4CAF50)
                    )

                    // Alertness
                    ClipClassificationRow(
                        label = "Alertness",
                        value = clipAlertness.label.replace("_", " "),
                        confidence = clipAlertnessConfidence,
                        color = Color(0xFFFF9800)
                    )

                    // Activity
                    ClipClassificationRow(
                        label = "Activity",
                        value = clipActivity.label.replace("_", " "),
                        confidence = clipActivityConfidence,
                        color = Color(0xFF2196F3)
                    )

                    // Safety (only show if not "none" or has high confidence)
                    if (clipSafety != NursingLabels.SafetyConcern.NONE || clipSafetyConfidence > 0.3f) {
                        ClipClassificationRow(
                            label = "Safety",
                            value = clipSafety.label.replace("_", " "),
                            confidence = clipSafetyConfidence,
                            color = if (clipSafety != NursingLabels.SafetyConcern.NONE)
                                Color(0xFFF44336) else Color(0xFF4CAF50)
                        )
                    }
                }
            }
        }

        // VLM Scene Analysis - at bottom for visibility
        lastObservation?.let { obs ->
            if (obs.chartNote.isNotBlank() && obs.chartNote != "VLM output could not be parsed") {
                Spacer(modifier = Modifier.height(8.dp))
                Surface(
                    color = Color.Black.copy(alpha = 0.85f),
                    shape = MaterialTheme.shapes.medium,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Row(
                            horizontalArrangement = Arrangement.SpaceBetween,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = "VLM Scene Analysis",
                                style = MaterialTheme.typography.labelMedium,
                                color = Color.Magenta,
                                fontWeight = FontWeight.Bold
                            )
                            if (obs.position != "unknown") {
                                Text(
                                    text = "${obs.position} • ${obs.alertness}",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color.Cyan
                                )
                            }
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = obs.chartNote,
                            style = MaterialTheme.typography.bodyMedium,
                            color = Color.White,
                            maxLines = 4
                        )
                    }
                }
            }
        }
    }
}

/**
 * Row component for displaying a CLIP classification result
 */
@Composable
fun ClipClassificationRow(
    label: String,
    value: String,
    confidence: Float,
    color: Color
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .padding(end = 4.dp)
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    drawCircle(color = color)
                }
            }
            Spacer(modifier = Modifier.width(4.dp))
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = Color.Gray
            )
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = value,
                style = MaterialTheme.typography.bodySmall,
                color = Color.White,
                fontWeight = FontWeight.Medium
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = "${(confidence * 100).toInt()}%",
                style = MaterialTheme.typography.labelSmall,
                color = color
            )
        }
    }
}

@Composable
fun ControlPanel(
    isMonitoring: Boolean,
    isAnalyzing: Boolean,
    secondsSinceMotion: Long,
    lastObservation: SlowPipeline.Observation?,
    useBackgroundService: Boolean = true,
    isServiceBound: Boolean = false,
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
                Column {
                    Text(
                        text = if (secondsSinceMotion > 60)
                            "Still for ${secondsSinceMotion / 60}m"
                        else
                            "Active",
                        style = MaterialTheme.typography.bodyMedium
                    )
                    if (useBackgroundService && isMonitoring) {
                        Text(
                            text = "Background service active",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }

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
fun SettingsScreen(
    useBackgroundService: Boolean = true,
    onBackgroundServiceToggle: (Boolean) -> Unit = {}
) {
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

        // Background service toggle
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(text = "Background Monitoring")
                Text(
                    text = "Continue monitoring when app is in background",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.outline
                )
            }
            Switch(
                checked = useBackgroundService,
                onCheckedChange = onBackgroundServiceToggle
            )
        }
        HorizontalDivider()

        SettingsItem(title = "VLM Analysis Interval", value = "2 seconds")
        SettingsItem(title = "Stillness Alert Threshold", value = "30 seconds")
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
