package run.leviathan.multitude

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Card
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import uniffi.leviathan_multitude_mobile.MultitudeException
import uniffi.leviathan_multitude_mobile.MultitudeNode
import uniffi.leviathan_multitude_mobile.PhoneActivity
import uniffi.leviathan_multitude_mobile.TickReport
import uniffi.leviathan_multitude_mobile.addressOf

private const val EMULATOR_HOST_RELAY = "http://10.0.2.2:8080"
private const val IDLE_POLL_MS = 2_000L
private const val ERROR_RETRY_MS = 5_000L
private const val NOT_TRAINING = "Not training"

private val Ink = Color(0xFF0E1113)
private val Bone = Color(0xFFE9E4D8)
private val Verdigris = Color(0xFF4FA89B)
private val Oxblood = Color(0xFFC0574E)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val darkBars = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT)
        enableEdgeToEdge(statusBarStyle = darkBars, navigationBarStyle = darkBars)
        val secret = PhoneKey.load(this)
        val device = DeviceReader(this)
        val statePath = filesDir.resolve("multitude.state").path
        setContent {
            MaterialTheme(
                colorScheme = darkColorScheme(
                    primary = Verdigris,
                    onPrimary = Ink,
                    background = Ink,
                    surface = Ink,
                    onBackground = Bone,
                    onSurface = Bone,
                    error = Oxblood,
                ),
            ) {
                Surface(Modifier.fillMaxSize()) {
                    MultitudeScreen(secret, statePath, device, ::keepScreenOn)
                }
            }
        }
    }

    private fun keepScreenOn(on: Boolean) {
        if (on) {
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        } else {
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }
}

@Composable
private fun MultitudeScreen(
    secret: ByteArray,
    statePath: String,
    device: DeviceReader,
    keepScreenOn: (Boolean) -> Unit,
) {
    val address = remember(secret) { addressOf(secret) }
    var relayUrl by rememberSaveable { mutableStateOf(EMULATOR_HOST_RELAY) }
    var joined by rememberSaveable { mutableStateOf(false) }
    var message by remember { mutableStateOf(NOT_TRAINING) }
    var report by remember { mutableStateOf<TickReport?>(null) }

    LaunchedEffect(joined, relayUrl) {
        keepScreenOn(joined)
        if (!joined) {
            message = NOT_TRAINING
            return@LaunchedEffect
        }
        message = "Joining $relayUrl"
        var node: MultitudeNode? = null
        try {
            while (isActive) {
                val next = try {
                    withContext(Dispatchers.Default) {
                        val current = node ?: MultitudeNode(relayUrl, secret, statePath).also { node = it }
                        current.tick(device.snapshot())
                    }
                } catch (error: MultitudeException) {
                    message = error.message ?: "The relay did not answer"
                    delay(ERROR_RETRY_MS)
                    continue
                }
                report = next
                message = next.message
                if (next.activity == PhoneActivity.CONVICTED) break
                if (next.activity != PhoneActivity.TRAINED) delay(IDLE_POLL_MS)
            }
        } finally {
            node?.close()
        }
    }

    Column(
        Modifier.fillMaxSize().safeDrawingPadding().padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                "Leviathan Multitude",
                style = MaterialTheme.typography.headlineMedium,
                fontFamily = FontFamily.Serif,
            )
            Text(
                "Train a shared language model while this phone charges.",
                style = MaterialTheme.typography.bodyLarge,
            )
        }
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("This phone", style = MaterialTheme.typography.labelMedium, color = Verdigris)
            Text(address, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall)
        }
        OutlinedTextField(
            value = relayUrl,
            onValueChange = { relayUrl = it.trim() },
            label = { Text("Relay") },
            singleLine = true,
            enabled = !joined,
            modifier = Modifier.fillMaxWidth(),
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Train on the charger", Modifier.weight(1f), style = MaterialTheme.typography.titleMedium)
            Switch(checked = joined, onCheckedChange = { joined = it })
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                val convicted = report?.activity == PhoneActivity.CONVICTED
                Text(
                    message,
                    color = if (convicted) MaterialTheme.colorScheme.error else LocalContentColor.current,
                )
                report?.let { ReportLines(it) }
            }
        }
        Text(
            "Keep the app open on the charger. Training pauses when the phone is unplugged, on mobile data, hot or low on battery.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

@Composable
private fun ReportLines(report: TickReport) {
    val round = report.round?.let { "Round $it" }
    val loss = report.globalLoss?.let { String.format(Locale.ROOT, "global loss %.3f", it) }
    val phones = report.phonesLastRound?.let { if (it == 1u) "1 phone last round" else "$it phones last round" }
    Text(
        listOfNotNull(round, loss, phones).joinToString(", "),
        fontFamily = FontFamily.Monospace,
        style = MaterialTheme.typography.bodyMedium,
    )
    Text(
        "Rounds counted ${report.roundsCounted}, audits answered ${report.auditsAnswered}",
        fontFamily = FontFamily.Monospace,
        style = MaterialTheme.typography.bodyMedium,
    )
}
