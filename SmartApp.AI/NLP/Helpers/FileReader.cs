﻿using System.Reflection;

namespace SmartBlazorApp.AI.NLP.Helpers;

public static class FileReader
{
    public static List<string> ReadFile(string filename)
    {
        List<string> result = new();

        using (StreamReader reader = new(filename))
        {
            string? line;

            while ((line = reader.ReadLine()) is not null)
            {
                if (!string.IsNullOrWhiteSpace(line))
                {
                    result.Add(line);
                }
            }
        }

        return result;
    }

    public static Stream? GetVocabularyStream()
    {
        string? vocabTextResourceStreamPath = Assembly.GetExecutingAssembly().GetManifestResourceNames()
            .Where(s => s.Contains("vocab")).FirstOrDefault();
        if (string.IsNullOrEmpty(vocabTextResourceStreamPath)) return null;
        return Assembly.GetExecutingAssembly().GetManifestResourceStream(vocabTextResourceStreamPath);
    }
}