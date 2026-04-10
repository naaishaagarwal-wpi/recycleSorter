package com.example.greetingcard

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.example.greetingcard.ui.theme.GreetingCardTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.compose.material3.AlertDialog
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.room.Room
import com.example.greetingcard.data.local.AppDatabase
import androidx.compose.runtime.*
import androidx.lifecycle.ViewModelProvider
import com.example.greetingcard.data.repository.UserRepository
import com.example.greetingcard.ui.viewmodel.UserViewModel
import com.example.greetingcard.ui.viewmodel.UserViewModelFactory
import androidx.core.net.toUri

class MainActivity : ComponentActivity() {

    private lateinit var db: AppDatabase
    private lateinit var userViewModel: UserViewModel

    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Create Room database
        db = Room.databaseBuilder(
            applicationContext,
            AppDatabase::class.java,
            "recycle_database"
        ).build()

        // Create repository
        val userRepository = UserRepository(db.userDao())

        // Create ViewModel using factory
        userViewModel = ViewModelProvider(
            this,
            UserViewModelFactory(userRepository)
        )[UserViewModel::class.java]

        setContent {
            GreetingCardTheme {
                RecycleAppWithDatabase(userViewModel)
            }
        }
    }
}

fun uriToBitmap(context: Context, uri: Uri): Bitmap? {
    return try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = ImageDecoder.createSource(context.contentResolver, uri)
            ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
                decoder.isMutableRequired = true
            }
        } else {
            @Suppress("DEPRECATION")
            MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
        }
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}


suspend fun classifyImageLocal(context: Context, imageUri: Uri): String {
    return withContext(Dispatchers.Default) {
        val bitmap = uriToBitmap(context, imageUri) ?: return@withContext "Failed to load image"

        val classifier = Classifier(context)
        val result = classifier.classify(bitmap)
        classifier.close()

        result
    }
}

enum class Screen {
    HOME,
    ACCOUNT,
    LANDING
}



@RequiresApi(Build.VERSION_CODES.O)
@Composable
fun RecycleAppWithDatabase(userViewModel: UserViewModel) {

    val user by userViewModel.user.collectAsState()

    var currentScreen by remember { mutableStateOf(Screen.HOME) }

    when (currentScreen) {
        Screen.HOME -> HomeScreen(
            points = user?.points ?: 0,
            onAccountClick = { currentScreen = Screen.ACCOUNT },
            onTryItOutClick = { currentScreen = Screen.LANDING }
        )

        Screen.ACCOUNT -> AccountScreen(
            name = user?.name ?: "",
            points = user?.points ?: 0,
            onNameChange = { newName ->
                user?.let {
                    userViewModel.updateName(newName)
                }
            },
            onBack = { currentScreen = Screen.HOME }
        )

        Screen.LANDING -> TryItOutScreen(
            onPointEarned = {
                // When user classifies an image, update points, streak, totalDaysUsed
                userViewModel.classifyImage(pointsToAdd = 1)
            },
            onBackHome = { currentScreen = Screen.HOME }
        )
    }
}

@Composable
fun HomeScreen(
    points: Int,
    onAccountClick: () -> Unit,
    onTryItOutClick: () -> Unit
) {
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color(0xFFEAF4D3) // softer eco background
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 28.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            // App Title
            Text(
                text = "Recycle Classifier",
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF2F5D2F)
            )

            Spacer(modifier = Modifier.height(12.dp))

            Text(
                text = "Snap a photo and find out if it's recyclable.",
                style = MaterialTheme.typography.bodyLarge,
                textAlign = TextAlign.Center,
                color = Color.DarkGray
            )

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = "Earn points every time you classify!",
                style = MaterialTheme.typography.bodyMedium,
                textAlign = TextAlign.Center,
                color = Color.Gray
            )

            Spacer(modifier = Modifier.height(32.dp))

            //Points Card
            Card(
                shape = RoundedCornerShape(20.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 6.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White)
            ) {
                Column(
                    modifier = Modifier
                        .padding(horizontal = 40.dp, vertical = 20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "Your Points",
                        style = MaterialTheme.typography.labelLarge,
                        color = Color.Gray
                    )

                    Text(
                        text = "$points",
                        style = MaterialTheme.typography.displaySmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF4CAF50)
                    )
                }
            }

            Spacer(modifier = Modifier.height(40.dp))

            // Try It Out Button
            Button(
                onClick = onTryItOutClick,
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(55.dp)
            ) {
                Text(
                    text = "Try It Out",
                    style = MaterialTheme.typography.titleMedium
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            //Account Button
            OutlinedButton(
                onClick = onAccountClick,
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(55.dp)
            ) {
                Text(
                    text = "Account",
                    style = MaterialTheme.typography.titleMedium
                )
            }
        }
    }
}


@Composable
fun AccountScreen(
    name: String,
    points: Int,
    onNameChange: (String) -> Unit,
    onBack: () -> Unit
) {
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color(0xFFEAF4D3)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 28.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Text(
                text = "Your Account",
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF2F5D2F)
            )

            Spacer(modifier = Modifier.height(32.dp))

            Card(
                shape = RoundedCornerShape(20.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 6.dp),
                colors = CardDefaults.cardColors(containerColor = Color.White)
            ) {
                Column(
                    modifier = Modifier
                        .padding(24.dp)
                        .fillMaxWidth(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {

                    OutlinedTextField(
                        value = name,
                        onValueChange = onNameChange,
                        label = { Text("Your name") },
                        singleLine = true,
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(24.dp))

                    Text(
                        text = "Points",
                        style = MaterialTheme.typography.labelLarge,
                        color = Color.Gray
                    )

                    Text(
                        text = "$points",
                        style = MaterialTheme.typography.displaySmall,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF4CAF50)
                    )
                }
            }

            Spacer(modifier = Modifier.height(40.dp))

            OutlinedButton(
                onClick = onBack,
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(55.dp)
            ) {
                Text(
                    text = "Back to Home",
                    style = MaterialTheme.typography.titleMedium
                )
            }
        }
    }
}


@Composable
fun TryItOutScreen(
    onPointEarned: () -> Unit,
    onBackHome: () -> Unit
) {
    val context = LocalContext.current
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }
    var classification by remember { mutableStateOf("") }
    var showLocations by remember { mutableStateOf(false) }

    val photoPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia(),
        onResult = { uri ->
            selectedImageUri = uri
            classification = ""
        }
    )


    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color(0xFFEAF4D3)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 28.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Spacer(modifier = Modifier.height(32.dp))

            Text(
                text = "Classify Waste",
                style = MaterialTheme.typography.headlineLarge,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF2F5D2F)
            )

            Spacer(modifier = Modifier.height(32.dp))

            Button(
                onClick = {
                    val resId = R.drawable.test_image
                    selectedImageUri =
                        "android.resource://${context.packageName}/$resId".toUri()

                    classification = ""
                },
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Use Test Image")
            }

            Spacer(modifier = Modifier.height(12.dp))

            OutlinedButton(
                onClick = {
                    photoPicker.launch(
                        PickVisualMediaRequest(
                            ActivityResultContracts.PickVisualMedia.ImageOnly
                        )
                    )
                },
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Upload Your Own Image")
            }

            Spacer(modifier = Modifier.height(24.dp))

            selectedImageUri?.let {

                Card(
                    shape = RoundedCornerShape(20.dp),
                    elevation = CardDefaults.cardElevation(6.dp),
                    colors = CardDefaults.cardColors(containerColor = Color.White)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {

                        AsyncImage(
                            model = it,
                            contentDescription = null,
                            modifier = Modifier
                                .height(250.dp)
                                .fillMaxWidth(),
                            contentScale = ContentScale.Crop
                        )

                        Spacer(modifier = Modifier.height(16.dp))

                        Button(
                            onClick = {
                                CoroutineScope(Dispatchers.IO).launch {
                                    val result =
                                        classifyImageLocal(context, it)
                                    withContext(Dispatchers.Main) {
                                        classification = result
                                        onPointEarned()
                                    }
                                }
                            },
                            shape = RoundedCornerShape(16.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Classify Image")
                        }
                    }
                }
            }

            if (classification.isNotEmpty()) {
                Spacer(modifier = Modifier.height(24.dp))

                Text(
                    text = "Result",
                    style = MaterialTheme.typography.labelLarge,
                    color = Color.Gray
                )

                Text(
                    text = classification,
                    style = MaterialTheme.typography.displaySmall,
                    fontWeight = FontWeight.Bold,
                    color = if (classification == "Recyclable")
                        Color(0xFF4CAF50)
                    else
                        Color(0xFFD32F2F)
                )

                Spacer(modifier = Modifier.height(12.dp))

                OutlinedButton(
                    onClick = { showLocations = true },
                    shape = RoundedCornerShape(16.dp)
                ) {
                    Text("See Locations")
                }
            }

            Spacer(modifier = Modifier.height(40.dp))

            OutlinedButton(
                onClick = onBackHome,
                shape = RoundedCornerShape(16.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Back to Home")
            }
        }
    }

    if (showLocations) {
        AlertDialog(
            onDismissRequest = { showLocations = false },
            confirmButton = {
                Button(onClick = { showLocations = false }) {
                    Text("Close")
                }
            },
            title = { Text("Nearby Locations") },
            text = {
                Text(
                    if (classification == "Recyclable")
                        "Metech Recycling\nRadius Recycling\nCurbside Computers"
                    else
                        "Superior Waste\nLocal Guys Trash Removal\nTrash Wizard"
                )
            }
        )
    }
}



@RequiresApi(Build.VERSION_CODES.O)
@Preview(showBackground = true)
@Composable
fun AppPreview() {
    GreetingCardTheme {
        // Use an in-memory database for preview
        val context = LocalContext.current
        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java).build()
        val repository = UserRepository(db.userDao())
        val viewModel = UserViewModel(repository)

        RecycleAppWithDatabase(userViewModel = viewModel)
    }
}

