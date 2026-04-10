package com.example.greetingcard.utils

import android.os.Build
import androidx.annotation.RequiresApi
import java.time.LocalDate
import java.time.format.DateTimeFormatter

object DateUtils {

    @RequiresApi(Build.VERSION_CODES.O)
    private val formatter = DateTimeFormatter.ISO_DATE

    @RequiresApi(Build.VERSION_CODES.O)
    fun today(): String {
        return LocalDate.now().format(formatter)
    }

    @RequiresApi(Build.VERSION_CODES.O)
    fun isYesterday(dateString: String): Boolean {
        val date = LocalDate.parse(dateString, formatter)
        return date.plusDays(1) == LocalDate.now()
    }

    @RequiresApi(Build.VERSION_CODES.O)
    fun isSameDay(dateString: String): Boolean {
        val date = LocalDate.parse(dateString, formatter)
        return date == LocalDate.now()
    }
}