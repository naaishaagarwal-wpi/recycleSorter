package com.example.greetingcard.ui.viewmodel

import android.os.Build
import androidx.annotation.RequiresApi
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.greetingcard.data.local.UserEntity
import com.example.greetingcard.data.repository.UserRepository
import com.example.greetingcard.utils.DateUtils
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import androidx.lifecycle.ViewModelProvider
class UserViewModel(private val repository: UserRepository) : ViewModel() {

    private val _user = MutableStateFlow<UserEntity?>(null)
    val user: StateFlow<UserEntity?> = _user.asStateFlow()

    init {
        viewModelScope.launch {
            repository.getUser().collect {
                _user.value = it
            }
        }
    }

    @RequiresApi(Build.VERSION_CODES.O)
    fun classifyImage(pointsToAdd: Int) {
        viewModelScope.launch {
            val currentUser = _user.value

            if (currentUser == null) {
                val newUser = UserEntity(
                    id = 1,
                    name = "Player",
                    points = pointsToAdd,
                    streak = 1,
                    totalDaysUsed = 1,
                    lastUsedDate = DateUtils.today()
                )
                repository.insertUser(newUser)
            } else {
                val today = DateUtils.today()

                var newStreak = currentUser.streak
                var newTotalDays = currentUser.totalDaysUsed

                if (DateUtils.isYesterday(currentUser.lastUsedDate)) {
                    newStreak += 1
                    newTotalDays += 1
                } else if (!DateUtils.isSameDay(currentUser.lastUsedDate)) {
                    newStreak = 1
                    newTotalDays += 1
                }

                val updatedUser = currentUser.copy(
                    points = currentUser.points + pointsToAdd,
                    streak = newStreak,
                    totalDaysUsed = newTotalDays,
                    lastUsedDate = today
                )

                repository.updateUser(updatedUser)
            }
        }
    }

    fun updateName(newName: String) {
        viewModelScope.launch {
            val currentUser = _user.value
            if (currentUser != null) {
                val updatedUser = currentUser.copy(name = newName)
                repository.updateUser(updatedUser)
            }
        }
    }
}

class UserViewModelFactory(
    private val repository: UserRepository
) : ViewModelProvider.Factory {

    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(UserViewModel::class.java)) {
            return UserViewModel(repository) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}